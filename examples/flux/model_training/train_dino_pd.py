import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from diffsynth.pipelines.flux_image_new import FluxImagePipeline
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, flux_parser
from diffsynth.models.lora import FluxLoRAConverter
from diffsynth.trainers.hf_url_dataset import HuggingFaceURLImageDataset

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration

from dino_noise import load_dino, find_dino_preserving_noise, latent_to_dino

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -------------------------
# Training module
# -------------------------
class DinoPDTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        dino_model_name="dinov2_vitl14_reg",
        dino_opt_steps=300,
        rollout_steps_min=5,
        rollout_steps_max=15,
        sigma_target_min=0.1,
        sigma_sampling="id_uniform",  # v7b: "sigma_uniform" samples uniform in sigma-space (inverse-CDF) to fix shift=3 skew
        dino_loss_layers=None,  # v8: list of 0-indexed ViT block indices, e.g. [4,11,23]; None = final-layer only (v7c behavior)
        **kwargs,  # swallow deprecated v5 args: max_sigma_gap_min/max, min_substep_sigma
    ):
        super().__init__()

        # Load FLUX pipeline
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = FluxImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        self.mapping_lora_state_dict = FluxLoRAConverter.align_to_diffsynth_format

        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.dino_opt_steps = dino_opt_steps
        self.rollout_steps_min = int(rollout_steps_min)
        self.rollout_steps_max = int(rollout_steps_max)
        self.sigma_target_min = float(sigma_target_min)
        self.sigma_sampling = str(sigma_sampling)
        self.dino_loss_layers = list(dino_loss_layers) if dino_loss_layers else None
        assert self.rollout_steps_max >= self.rollout_steps_min >= 1
        assert 0.0 <= self.sigma_target_min < 1.0
        assert self.sigma_sampling in ("id_uniform", "sigma_uniform")

        for k in ["max_sigma_gap_min", "max_sigma_gap_max", "min_substep_sigma"]:
            if k in kwargs:
                print(f"[v6 warning] Argument --{k} is deprecated and will be ignored.")

        # Load DINOv2 on CPU; will be moved to the correct device in set_accelerator.
        # Stored outside nn.Module registry to keep DDP away from it.
        dino = load_dino(model_name=dino_model_name, device="cpu")
        object.__setattr__(self, "_dino", dino)

        self.accelerator = None

    def set_accelerator(self, accelerator: Accelerator):
        self.accelerator = accelerator
        # Move DINOv2 to the process device now that accelerator is ready.
        self._dino.to(accelerator.device)

    # ------------------------------------------------------------------
    # Pipeline preprocessing (identical to the wavelet training module)
    # ------------------------------------------------------------------
    def forward_preprocess(self, data):
        inputs_posi   = {"prompt": data["prompt"]}
        inputs_nega   = {"negative_prompt": ""}
        inputs_shared = {
            "input_image": data["image"],
            "height":      data["image"].size[1],
            "width":       data["image"].size[0],
            "cfg_scale":   1,
            "embedded_guidance": 1,  # v7-c: revert to v6 setup. v7-α/b's embed=3.5 trained the LoRA to cancel embed_g's detail contribution → blur. Inference uses embed=3.5; the structural mismatch is accepted as the smaller cost.
            "t5_sequence_length": 512,
            "tiled":       False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }

        for extra_input in self.extra_inputs:
            inputs_shared[extra_input] = data[extra_input]

        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )
        return {**inputs_shared, **inputs_posi}

    # ------------------------------------------------------------------
    # Forward: rollout-aware rectified flow matching (v6 — fixed sigma=1.0 start).
    #
    # v6 Rationale:
    # v5 sampled sigma_start and seeded with fresh Adam-optimized z_t*. However,
    # at inference, any latent at sigma < 1.0 is reached via an Euler rollout
    # from sigma=1.0. These drifted latents have a different distribution than
    # the fresh z_t* seeds. v6 fixes sigma_start=1.0 to ensure the training
    # starting distribution matches inference, following the same path to sigma_target.
    #
    # Per-sample algorithm:
    #   1. Fix start timestep_id_start = 0 (sigma = 1.0).
    #   2. Sample target timestep_id_target (j > 0) with sigma_j >= sigma_target_min.
    #   3. Seed z at sigma=1.0 using DINO-preserving noise z1*.
    #   4. Sample K ~ randint(rollout_steps_min, rollout_steps_max + 1).
    #   5. Run K detached Euler steps from 0 to j to produce z_rolled at sigma_j.
    #   6. Compute flow-matching loss at timestep_j with rectified target to z0.
    # ------------------------------------------------------------------
    def forward(self, data, inputs=None, step: int | None = None):
        if inputs is None:
            inputs = self.forward_preprocess(data)

        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        dtype, device = self.pipe.torch_dtype, self.pipe.device

        z0 = inputs["input_latents"].float()  # (1, 16, H, W)

        self.pipe.scheduler.set_timesteps(self.pipe.scheduler.num_train_timesteps, training=True)
        N = self.pipe.scheduler.num_train_timesteps
        sigmas = self.pipe.scheduler.sigmas  # monotonically decreasing in j

        # v6: fixed start at sigma=1.0 (timestep_id=0)
        timestep_id_start = 0
        sigma_start = sigmas[timestep_id_start].item()

        # Sample target timestep_id_target (j > 0) such that sigmas[j] >= sigma_target_min.
        # sigma_target_min=0.1 is usually near j=900 for N=1000.
        valid_upper = N - 1
        while valid_upper > 0 and sigmas[valid_upper].item() < self.sigma_target_min:
            valid_upper -= 1

        if self.sigma_sampling == "sigma_uniform":
            # v7b: sample sigma_target uniform in sigma-space, then snap to nearest id.
            # Counters FLUX shift=3 right-skew of id-uniform sampling (25% of samples in [0.9,1.0],
            # only 4% in [0.1,0.2]). Cap upper at sigmas[1] to keep id_target>=1.
            sigma_lo = float(self.sigma_target_min)
            sigma_hi = float(sigmas[1].item())
            sampled_sigma = float(np.random.uniform(sigma_lo, sigma_hi))
            valid_sigmas = sigmas[1:valid_upper + 1]
            timestep_id_target = int(torch.argmin(torch.abs(valid_sigmas - sampled_sigma)).item()) + 1
        else:
            timestep_id_target = torch.randint(1, valid_upper + 1, (1,)).item()
        
        # Sample K steps for the detached rollout
        K = int(np.random.randint(self.rollout_steps_min, self.rollout_steps_max + 1))
        # can't subdivide finer than 1 per native timestep
        K_eff = min(K, timestep_id_target - timestep_id_start)

        timestep_target = self.pipe.scheduler.timesteps[torch.tensor([timestep_id_target])].to(dtype=dtype, device=device)
        sigma_target = self.pipe.scheduler.sigmas[timestep_id_target].item()

        # Find z_1* that preserves DINOv2 features of z0 at t=1.0
        with torch.enable_grad():
            z1_star, dino_dist_start = find_dino_preserving_noise(
                z0=z0,
                t=1.0,
                vae_decoder=self.pipe.vae_decoder,
                dino=self._dino,
                n_steps=self.dino_opt_steps,
                loss_layers=self.dino_loss_layers,
            )

        z1_star = z1_star.detach()
        # v6: start is always sigma=1.0, so z_start = z1_star
        z_curr = z1_star

        # Temporary noise injection purely to condition model_fn for the offline rollout predictions
        inputs["noise"] = z1_star.to(dtype=dtype, device=device)
        inputs["input_latents"] = z0.to(dtype=dtype, device=device)

        # K_eff-step detached Euler rollout along equally-spaced indices from 0 to timestep_id_target.
        rollout_ids = np.linspace(
            timestep_id_start, timestep_id_target, K_eff + 1
        ).round().astype(int).tolist()
        # dedupe while preserving order
        seen = set(); rollout_ids = [x for x in rollout_ids if not (x in seen or seen.add(x))]

        with torch.no_grad():
            for k in range(len(rollout_ids) - 1):
                id_k = rollout_ids[k]
                id_next = rollout_ids[k + 1]
                sigma_k = self.pipe.scheduler.sigmas[id_k].item()
                sigma_next = self.pipe.scheduler.sigmas[id_next].item()
                timestep_k = self.pipe.scheduler.timesteps[torch.tensor([id_k])].to(dtype=dtype, device=device)
                
                # refresh noise field to match current z (keeps model_fn conditioning consistent)
                t_safe_k = max(sigma_k, 1e-4)
                z1_k = (z_curr.float() - (1.0 - t_safe_k) * z0) / t_safe_k
                inputs["noise"] = z1_k.to(dtype=dtype, device=device)
                inputs["latents"] = z_curr.to(dtype=dtype, device=device)
                v_k = self.pipe.model_fn(**models, **inputs, timestep=timestep_k)
                z_curr = z_curr + (sigma_next - sigma_k) * v_k

        z_target = z_curr  # at sigma_target after K_eff detached Euler steps

        # Rectified target at sigma_target pointing precisely to z0
        t_safe_target = max(sigma_target, 1e-4)
        z1_target = (z_target.detach().float() - (1.0 - t_safe_target) * z0) / t_safe_target
        inputs["noise"] = z1_target.to(dtype=dtype, device=device)

        # Compute flow-matching loss at the chosen timestep (grad step)
        inputs["latents"]        = z_target.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        training_target          = self.pipe.scheduler.training_target(
            inputs["input_latents"], inputs["noise"], timestep_target
        )
        noise_pred = self.pipe.model_fn(**models, **inputs, timestep=timestep_target)
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep_target)

        # --- Log stats ----------------------------------------------------
        if self.accelerator is not None and step is not None:
            with torch.no_grad():
                stats = self.accelerator.gather(
                    torch.stack([
                        torch.tensor(dino_dist_start, device=device, dtype=torch.float32),
                        loss.detach().float(),
                        torch.tensor(float(sigma_start), device=device, dtype=torch.float32),
                        torch.tensor(float(sigma_target), device=device, dtype=torch.float32),
                        torch.tensor(float(len(rollout_ids) - 1), device=device, dtype=torch.float32),
                    ])[None]
                ).mean(dim=0)

            if self.accelerator.is_main_process:
                self.accelerator.log(
                    {
                        "dino/distance_start": stats[0].item(),
                        "loss":                stats[1].item(),
                        "rollout/sigma_start": stats[2].item(),
                        "rollout/sigma_target": stats[3].item(),
                        "rollout/sigma_gap":    stats[2].item() - stats[3].item(),
                        "rollout/num_steps":    stats[4].item(),
                    },
                    step=step,
                )
        return loss

    @torch.no_grad()
    def run_validation(
        self,
        step: int,
        output_dir: str,
        val_image_path: str = "models/ppd/test1.jpg",
        val_prompt: str = "A photorealistic scene. High resolution, realistic textures.",
        height: int = 704,
        width: int = 1280,
        cfg_scale: float = 1.0,  # v7-α: cfg=1 matches training (no CFG at inference)
        num_inference_steps: int = 50,
        dino_opt_steps: int | None = None,
        save_frames_at=(0, 10, 20, 30, 40, 49),
    ):
        """Inline validation: mirrors validate_dino_trajectory.py but reuses pipe+DINO.
        Call on main process only. Returns a dict of scalars."""
        os.makedirs(output_dir, exist_ok=True)
        device = self.pipe.device
        was_training = self.pipe.dit.training
        self.pipe.dit.eval()
        # Snapshot scheduler state so pipe(...) call below doesn't corrupt training mode
        sched = self.pipe.scheduler
        prev_sched_state = {
            "sigmas": sched.sigmas.clone() if hasattr(sched, "sigmas") else None,
            "timesteps": sched.timesteps.clone() if hasattr(sched, "timesteps") else None,
            "training": getattr(sched, "training", False),
            "linear_timesteps_weights": getattr(sched, "linear_timesteps_weights", None),
        }
        try:
            img_pil = Image.open(val_image_path).convert("RGB").resize((width, height), resample=Image.LANCZOS)
            img_pil.save(os.path.join(output_dir, "input.png"))

            img_t = self.pipe.preprocess_image(img_pil).to(device=device, dtype=self.pipe.torch_dtype)
            z0 = self.pipe.vae_encoder(img_t, tiled=False)
            z0_f32 = z0.float()
            target_feats = latent_to_dino(self.pipe.vae_decoder, self._dino, z0_f32)
            target_feats = {k: v.detach() for k, v in target_feats.items()}

            with torch.enable_grad():
                z1_star, _ = find_dino_preserving_noise(
                    z0=z0_f32, t=1.0,
                    vae_decoder=self.pipe.vae_decoder, dino=self._dino,
                    n_steps=dino_opt_steps if dino_opt_steps is not None else self.dino_opt_steps,
                    loss_layers=self.dino_loss_layers,
                )
            noise = z1_star.to(dtype=self.pipe.torch_dtype).contiguous()

            step_indices, dino_distances, cls_distances, sigma_vals = [], [], [], []
            save_steps_set = set(save_frames_at)
            original_step = self.pipe.scheduler.step
            step_counter = [0]

            def step_hook(model_output, timestep, sample, **kwargs):
                new_latents = original_step(model_output, timestep, sample, **kwargs)
                idx = step_counter[0]; step_counter[0] += 1
                sigma = (timestep.cpu().item() / self.pipe.scheduler.num_train_timesteps)
                cur_feats = latent_to_dino(self.pipe.vae_decoder, self._dino, new_latents.float())
                patch_d = (1.0 - F.cosine_similarity(cur_feats["patches"], target_feats["patches"], dim=-1).mean()).item()
                cls_d   = (1.0 - F.cosine_similarity(cur_feats["cls"],     target_feats["cls"],     dim=-1).mean()).item()
                # VGGT consumes x_norm_patchtokens only — headline metric is patch distance.
                total_d = patch_d
                step_indices.append(idx); dino_distances.append(total_d)
                cls_distances.append(cls_d); sigma_vals.append(sigma)
                if idx in save_steps_set:
                    img = self.pipe.vae_decoder(new_latents.to(self.pipe.torch_dtype), tiled=False).float()
                    img = (img.clamp(-1, 1) + 1) / 2
                    img = img.squeeze(0).permute(1, 2, 0)
                    Image.fromarray((img * 255).byte().cpu().numpy()).save(
                        os.path.join(output_dir, f"step_{idx:03d}_sigma{sigma:.3f}.png"))
                return new_latents

            self.pipe.scheduler.step = step_hook
            try:
                out = self.pipe(
                    prompt=val_prompt, height=height, width=width,
                    cfg_scale=cfg_scale, num_inference_steps=num_inference_steps,
                    noise=noise,
                )
            finally:
                self.pipe.scheduler.step = original_step

            out.save(os.path.join(output_dir, "output_final.png"))

            final_d = dino_distances[-1]; peak_d = max(dino_distances); min_d = min(dino_distances)
            peak_step = step_indices[dino_distances.index(peak_d)]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(step_indices, dino_distances, marker=".", linewidth=1.5)
            axes[0].set_xlabel("Denoising step"); axes[0].set_ylabel("DINO distance to z0")
            axes[0].set_title(f"step {step} — final={final_d:.3f} peak={peak_d:.3f}@{peak_step}")
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(sigma_vals, dino_distances, marker=".", linewidth=1.5, color="tab:orange")
            axes[1].invert_xaxis(); axes[1].set_xlabel("sigma  (1→0)")
            axes[1].set_ylabel("DINO distance to z0"); axes[1].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "dino_trajectory.png"), dpi=120)
            plt.close(fig)

            metrics = {
                "val/dino_final": final_d, "val/dino_peak": peak_d, "val/dino_min": min_d,
                "val/dino_peak_step": float(peak_step),
                "val/cls_final": cls_distances[-1], "val/cls_peak": max(cls_distances),
            }
            for i in save_frames_at:
                if i < len(dino_distances):
                    metrics[f"val/dino_step_{i:02d}"] = dino_distances[i]
            return metrics
        finally:
            if was_training:
                self.pipe.dit.train()
            # Restore scheduler to training mode so next training step's
            # forward_preprocess sees scheduler.training=True (otherwise
            # FluxImageUnit_InputImageEmbedder returns input_latents=None).
            if prev_sched_state["sigmas"] is not None:
                sched.sigmas = prev_sched_state["sigmas"]
                sched.timesteps = prev_sched_state["timesteps"]
            sched.training = prev_sched_state["training"]
            if prev_sched_state["linear_timesteps_weights"] is not None:
                sched.linear_timesteps_weights = prev_sched_state["linear_timesteps_weights"]


# -------------------------
# Training loop
# -------------------------
def launch_training_task(
    dataset, model, model_logger,
    learning_rate=1e-5, weight_decay=1e-2, num_workers=8,
    save_steps=None, num_epochs=1, gradient_accumulation_steps=1,
    find_unused_parameters=False, args=None,
):
    if args is not None:
        learning_rate             = args.learning_rate
        weight_decay              = args.weight_decay
        num_workers               = args.dataset_num_workers
        save_steps                = args.save_steps
        num_epochs                = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters    = args.find_unused_parameters

    optimizer  = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler  = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers
    )
    config      = ProjectConfiguration(
        project_dir=args.output_path,
        logging_dir=os.path.join(args.output_path, "logs"),
    )
    accelerator = Accelerator(
        log_with="tensorboard",
        project_config=config,
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    accelerator.init_trackers(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    raw_model = accelerator.unwrap_model(model)
    if hasattr(model, "module") and hasattr(model.module, "set_accelerator"):
        model.module.set_accelerator(accelerator)
    elif hasattr(raw_model, "set_accelerator"):
        raw_model.set_accelerator(accelerator)

    global_step = 0
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data, step=global_step)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                # Inline validation after each checkpoint save
                if (save_steps is not None
                        and model_logger.num_steps % save_steps == 0
                        and getattr(args, "val_enabled", False)):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        with open(args.val_prompt_file) as f:
                            val_prompt = f.read().strip()
                        val_dir = os.path.join(args.output_path, "val", f"step-{model_logger.num_steps}")
                        metrics = raw_model.run_validation(
                            step=model_logger.num_steps,
                            output_dir=val_dir,
                            val_image_path=args.val_image,
                            val_prompt=val_prompt,
                            height=args.val_height,
                            width=args.val_width,
                            cfg_scale=args.val_cfg_scale,
                            num_inference_steps=args.val_num_inference_steps,
                            dino_opt_steps=args.val_dino_opt_steps,
                        )
                        accelerator.log(metrics, step=model_logger.num_steps)
                        print(f"[val step-{model_logger.num_steps}] " +
                              " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                    accelerator.wait_for_everyone()
                scheduler.step()
                global_step += 1

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = flux_parser()
    parser.add_argument("--dino_model_name", type=str, default="dinov2_vitl14_reg",
                        help="DINOv2 model name (torch.hub facebookresearch/dinov2).")
    parser.add_argument("--dino_opt_steps", type=int, default=300,
                        help="Adam steps per training sample to find z_t*.")
    parser.add_argument("--rollout_steps_min", type=int, default=5,
                        help="Lower bound for per-step sampled rollout_steps ~ randint(min, max+1). Default 5 for v6.")
    parser.add_argument("--rollout_steps_max", type=int, default=15,
                        help="Upper bound for per-step sampled rollout_steps ~ randint(min, max+1). Default 15 for v6.")
    parser.add_argument("--sigma_target_min", type=float, default=0.1,
                        help="Floor on sigma_target to avoid 1/sigma blow-up of the rectified v-target when drift is amplified at small sigma.")
    parser.add_argument("--dino_loss_layers", type=int, nargs="+", default=None,
                        help="v8: 0-indexed ViT block indices for multi-layer DINO loss "
                             "(e.g. --dino_loss_layers 4 11 23). Default None = final-layer only (v7c).")
    parser.add_argument("--sigma_sampling", type=str, default="id_uniform",
                        choices=["id_uniform", "sigma_uniform"],
                        help="v7b: 'sigma_uniform' samples sigma_target uniformly in sigma-space (inverse-CDF over the timestep grid) to counter FLUX shift=3 right-skew. Default 'id_uniform' matches v6/v7a.")
    
    # Deprecated v5 arguments (ignored in v6)
    parser.add_argument("--max_sigma_gap_min", type=float, default=0.0, help="[v6 ignored]")
    parser.add_argument("--max_sigma_gap_max", type=float, default=0.0, help="[v6 ignored]")
    parser.add_argument("--min_substep_sigma", type=float, default=0.0, help="[v6 ignored]")

    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum dataset samples (for debugging).")
    parser.add_argument("--val_enabled", type=lambda s: s.lower() not in ("0","false","no"), default=True,
                        help="Run inline validation after each checkpoint save (default: True).")
    parser.add_argument("--val_image", type=str, default="models/ppd/test1.jpg")
    parser.add_argument("--val_prompt_file", type=str, default="models/ppd/test1.txt",
                        help="Text file whose contents are used as the validation prompt.")
    parser.add_argument("--val_height", type=int, default=704)
    parser.add_argument("--val_width", type=int, default=1280)
    parser.add_argument("--val_cfg_scale", type=float, default=1.0,
                        help="v7-α default: cfg=1 matches training (no CFG at inference).")
    parser.add_argument("--val_num_inference_steps", type=int, default=50)
    parser.add_argument("--val_dino_opt_steps", type=int, default=300)
    args = parser.parse_args()

    dataset = HuggingFaceURLImageDataset(
        dataset_name="bghira/photo-concept-bucket",
        url_field="url",
        text_field="cogvlm_caption",
        cache_dir="./data/images",
        max_pixels=args.max_pixels,
        height=None,
        width=None,
        data_file_keys=("image", "prompt"),
        repeat=args.dataset_repeat,
        max_samples=args.max_samples,
    )

    print(f"Dataset size: {len(dataset)}")

    model = DinoPDTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        dino_model_name=args.dino_model_name,
        dino_opt_steps=args.dino_opt_steps,
        rollout_steps_min=args.rollout_steps_min,
        rollout_steps_max=args.rollout_steps_max,
        sigma_target_min=args.sigma_target_min,
        sigma_sampling=args.sigma_sampling,
        dino_loss_layers=args.dino_loss_layers,
        # Pass deprecated args to trigger warning
        max_sigma_gap_min=args.max_sigma_gap_min,
        max_sigma_gap_max=args.max_sigma_gap_max,
        min_substep_sigma=args.min_substep_sigma,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        state_dict_converter=FluxLoRAConverter.align_to_opensource_format
            if args.align_to_opensource_format else lambda x: x,
    )
    launch_training_task(dataset, model, model_logger, args=args, num_workers=2)
