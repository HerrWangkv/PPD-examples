import os
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from diffsynth.pipelines.flux_image_new import FluxImagePipeline
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, flux_parser
from diffsynth.models.lora import FluxLoRAConverter
from diffsynth.trainers.hf_url_dataset import HuggingFaceURLImageDataset

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration

from dino_noise import load_dino, find_dino_preserving_noise

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
        max_sigma_gap=0.1,
        rollout_steps=5,
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
        self.max_sigma_gap = max_sigma_gap
        self.rollout_steps = rollout_steps

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
            "embedded_guidance": 1,
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
    # Forward: rollout-aware rectified flow matching (v5 — K-step rollout).
    #
    # Per-sample algorithm:
    #   1. Sample start timestep_id_start (i) and target timestep_id_target (j > i)
    #      with (sigma_i - sigma_j) <= max_sigma_gap.
    #   2. Seed z at sigma_i using DINO-preserving noise z_{t_i}*.
    #   3. Run K detached Euler steps along equally-spaced indices between i and j
    #      to produce z_rolled at sigma_j. K=1 recovers v4's 1-step jump.
    #   4. Compute flow-matching loss at timestep_j with rectified target to z0.
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

        # Sample timestep_id_start, then clamp the target so that
        # (sigma_start - sigma_target) <= max_sigma_gap. This avoids
        # rectified-target blow-up when the Euler jump is long.
        timestep_id_start = torch.randint(0, N - 1, (1,)).item()
        sigma_start_val = sigmas[timestep_id_start].item()
        sigma_floor = sigma_start_val - self.max_sigma_gap
        # largest j > i with sigmas[j] >= sigma_floor
        max_target = timestep_id_start + 1
        while max_target + 1 < N and sigmas[max_target + 1].item() >= sigma_floor:
            max_target += 1
        timestep_id_target = torch.randint(timestep_id_start + 1, max_target + 1, (1,)).item()
        timestep_id_start = torch.tensor([timestep_id_start])
        timestep_id_target = torch.tensor([timestep_id_target])

        timestep_target = self.pipe.scheduler.timesteps[timestep_id_target].to(dtype=dtype, device=device)

        # sigma = t in [0, 1] for FlowMatchScheduler
        sigma_start = self.pipe.scheduler.sigmas[timestep_id_start].item()
        sigma_target = self.pipe.scheduler.sigmas[timestep_id_target].item()

        # Find z_t* that preserves DINOv2 features of z0 at t_start
        with torch.enable_grad():
            z_t_star, dino_dist_start = find_dino_preserving_noise(
                z0=z0,
                t=sigma_start,
                vae_decoder=self.pipe.vae_decoder,
                dino=self._dino,
                n_steps=self.dino_opt_steps,
            )

        z_t_star = z_t_star.detach()
        t_safe_start = max(sigma_start, 1e-4)

        # Temporary noise injection purely to condition model_fn for the offline rollout predictions
        z1_star = (z_t_star - (1.0 - t_safe_start) * z0) / t_safe_start
        inputs["noise"] = z1_star.to(dtype=dtype, device=device)
        inputs["input_latents"] = z0.to(dtype=dtype, device=device)

        # K-step detached Euler rollout along equally-spaced indices from i to j.
        # K=1 collapses to v4 (single jump from sigma_start directly to sigma_target).
        K = max(1, int(self.rollout_steps))
        span = timestep_id_target.item() - timestep_id_start.item()
        K_eff = min(K, span)  # can't subdivide finer than 1 per native timestep
        rollout_ids = np.linspace(
            timestep_id_start.item(), timestep_id_target.item(), K_eff + 1
        ).round().astype(int).tolist()
        # dedupe while preserving order
        seen = set(); rollout_ids = [x for x in rollout_ids if not (x in seen or seen.add(x))]

        z_curr = z_t_star
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

        z_target = z_curr  # at sigma_target after K detached Euler steps

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


# -------------------------
# Training loop  (identical structure to train.py)
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
    parser.add_argument("--max_sigma_gap", type=float, default=0.1,
                        help="Max (sigma_start - sigma_target) for the detached Euler rollout.")
    parser.add_argument("--rollout_steps", type=int, default=5,
                        help="Number of detached Euler steps between sigma_start and sigma_target. K=1 = v4 behavior.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum dataset samples (for debugging).")
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
        max_sigma_gap=args.max_sigma_gap,
        rollout_steps=args.rollout_steps,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        state_dict_converter=FluxLoRAConverter.align_to_opensource_format
            if args.align_to_opensource_format else lambda x: x,
    )
    launch_training_task(dataset, model, model_logger, args=args, num_workers=2)
