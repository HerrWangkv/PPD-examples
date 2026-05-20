import os, time
from pathlib import Path
from datetime import datetime

import math
import numpy as np
import torch
from tqdm import tqdm

from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ControlNetInput
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, flux_parser
from diffsynth.models.lora import FluxLoRAConverter
from diffsynth.trainers.hf_url_dataset import HuggingFaceURLImageDataset

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration

from wavelet_noise import generate_wavelet_structured_noise_batch_vectorized

os.environ["TOKENIZERS_PARALLELISM"] = "false"



# -------------------------
# Training module
# -------------------------
class FluxTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
    ):
        super().__init__()

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = FluxImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        self.mapping_lora_state_dict = FluxLoRAConverter.align_to_diffsynth_format

        # Switch to training mode (LoRA / trainable modules)
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        # Store configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []

        # Accelerator will be injected after accelerator.prepare(...)
        self.accelerator: Accelerator | None = None

    def set_accelerator(self, accelerator: Accelerator):
        self.accelerator = accelerator

    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "embedded_guidance": 1,
            "t5_sequence_length": 512,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }

        # Extra inputs
        controlnet_input = {}
        for extra_input in self.extra_inputs:
            if extra_input.startswith("controlnet_"):
                controlnet_input[extra_input.replace("controlnet_", "")] = data[extra_input]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if len(controlnet_input) > 0:
            inputs_shared["controlnet_inputs"] = [ControlNetInput(**controlnet_input)]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None, step: int | None = None):
        if inputs is None:
            inputs = self.forward_preprocess(data)

        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}

        input_latents = inputs["input_latents"]
        h, w = input_latents.shape[-2:]

        # Sample params
        radius = 4 + np.random.exponential(scale=16)
        radius = min(radius, min(h, w) // 2)

        # auto J from radius
        nyquist = min(h, w) / 2.0
        f_min = max(min(float(radius), nyquist) / nyquist, 1e-2)
        auto_J = max(1, math.ceil(-math.log2(f_min)))

        # Hard upper bound: LL must stay >= 2px → J <= log2(min(h,w))
        j_max_latent = int(math.log2(min(h, w)))

        drop_ll = bool(np.random.random() < 0.8)
        j_min = max(auto_J, 3)
        j_max = min(auto_J + 4, j_max_latent)
        # If no valid J exists (auto_J already near latent limit), skip drop_ll
        if drop_ll and j_min <= j_max:
            J = int(np.random.randint(j_min, j_max + 1))
        else:
            drop_ll = False
            J = None

        input_noise = torch.randn_like(input_latents.float())

        structured_noise = generate_wavelet_structured_noise_batch_vectorized(
            input_latents.float(),
            radius_map=radius,
            drop_ll=drop_ll,
            J=J,
            input_noise=input_noise,
        )

        inputs["noise"] = structured_noise.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        loss = self.pipe.training_loss(**models, **inputs)
        
        # Log stats (only if accelerator injected)
        if self.accelerator is not None and step is not None:
            with torch.no_grad():
                n = structured_noise.detach().float()
                stats = torch.stack([n.mean(), n.std(), n.abs().max(), loss.detach().float()])  # (4,)
                stats = self.accelerator.gather(stats[None]).mean(dim=0)  # (4,)

            if self.accelerator.is_main_process:
                self.accelerator.log(
                    {
                        "noise/mean": stats[0].item(),
                        "noise/std": stats[1].item(),
                        "noise/max_abs": stats[2].item(),
                        "noise/radius": float(radius),
                        "noise/drop_ll": float(drop_ll),
                        "noise/J": float(J) if J is not None else float(auto_J),
                        "loss": stats[3].item(),
                    },
                    step=step,
                )
        return loss


# -------------------------
# Training loop
# -------------------------
def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    args=None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    config = ProjectConfiguration(project_dir=args.output_path, logging_dir=os.path.join(args.output_path, "logs"))
    accelerator = Accelerator(
        log_with="tensorboard",
        project_config=config,
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    accelerator.init_trackers(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    # Inject accelerator into model (so forward() can log)
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
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use for debugging.")
    args = parser.parse_args()

    dataset = HuggingFaceURLImageDataset(
        dataset_name="bghira/photo-concept-bucket",
        url_field="url",
        text_field="cogvlm_caption",  # or "description", "alt", "title"
        cache_dir="./data/images",
        max_pixels=args.max_pixels,
        height=None,  # Use dynamic resolution
        width=None,
        data_file_keys=("image", "prompt"),  # Return both image and prompt
        repeat=args.dataset_repeat,
        max_samples=args.max_samples,  # Limit to 100 samples for debugging (set to None for full dataset)
    )
    
    print(f"Dataset size: {len(dataset)}")
    model = FluxTrainingModule(
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
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        state_dict_converter=FluxLoRAConverter.align_to_opensource_format if args.align_to_opensource_format else lambda x:x,
    )
    launch_training_task(dataset, model, model_logger, args=args, num_workers=2)
