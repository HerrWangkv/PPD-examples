import os
import sys
import glob
import cv2
import time
import random
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ControlNetInput
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, flux_parser
from diffsynth.models.lora import FluxLoRAConverter
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from wavelet_noise import generate_wavelet_structured_noise_batch_vectorized

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------
# Custom OpenScene Dataset (Balanced Sampling)
# -------------------------
class OpenSceneDataset(Dataset):
    def __init__(self, root_dir, height=704, width=1280, prompt="A photorealistic driving scene in a city, view from a car dashboard. Natural lighting, urban buildings, trees, cars on the street. High resolution, realistic textures.", max_samples=None):
        self.root_dir = root_dir
        self.height = height
        self.width = width
        self.prompt = prompt
        
        print(f"Scanning dataset at {root_dir}...")
        
        self.labeled_pairs = []
        self.unlabeled_images = []
        
        # 1. Find all clip-level 'images' folders
        # Structure: root_dir / log_id / clip_id / images
        clip_image_folders = sorted(glob.glob(os.path.join(root_dir, "*", "*", "images")))
        
        print(f"Found {len(clip_image_folders)} clips. Categorizing based on '_DONE' flag...")
        
        # 2. Iterate Clips (Not Images) to minimize I/O
        for img_dir in tqdm(clip_image_folders, desc="Scanning Clips"):
            # img_dir: .../log/clip/images
            clip_root = os.path.dirname(img_dir)
            disp_dir = os.path.join(clip_root, "disparity")
            done_flag = os.path.join(disp_dir, "_DONE")
            # Single I/O check per clip
            is_labeled = os.path.exists(done_flag)
            
            filenames = os.listdir(img_dir)
            for fname in filenames:
                if not fname.endswith(".jpg"): continue
                
                jpg_path = os.path.join(img_dir, fname)
                
                if is_labeled:
                    # Infer disparity path deterministically
                    # .../disparity/filename.png
                    png_name = fname.replace(".jpg", ".png")
                    png_path = os.path.join(disp_dir, png_name)
                    self.labeled_pairs.append((jpg_path, png_path))
                else:
                    self.unlabeled_images.append(jpg_path)
        
        print(f"Dataset Summary:")
        print(f"  - Labeled (with depth): {len(self.labeled_pairs)}")
        print(f"  - Unlabeled (no depth): {len(self.unlabeled_images)}")
        print(f"  - Target Resolution: {self.width}x{self.height}")
        
        self.virtual_length = max(len(self.labeled_pairs), len(self.unlabeled_images))
        if max_samples:
            self.virtual_length = min(self.virtual_length, max_samples)

    def __len__(self):
        return self.virtual_length

    def __getitem__(self, idx):
        # Balanced Sampling (50/50)
        use_labeled = random.random() < 0.5
        if len(self.labeled_pairs) == 0: use_labeled = False
        if len(self.unlabeled_images) == 0: use_labeled = True

        path_img = None
        path_disp = None
        has_depth = False

        if use_labeled:
            path_img, path_disp = random.choice(self.labeled_pairs)
            has_depth = True
        else:
            path_img = random.choice(self.unlabeled_images)
            has_depth = False

        image = Image.open(path_img).convert("RGB")
        image = image.resize((self.width, self.height), Image.LANCZOS)
        
        if has_depth:
            # Load 16-bit PNG
            disp_16bit = cv2.imread(path_disp, cv2.IMREAD_UNCHANGED)
            if disp_16bit is None:
                raise ValueError("Read error")
            disp_16bit = cv2.resize(disp_16bit, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            # Normalize 0~65535 -> 0.0~1.0
            disp_float = disp_16bit.astype(np.float32) / 65535.0
            # Shape: (1, H, W)
            disp_tensor = torch.from_numpy(disp_float).unsqueeze(0)
        else:
            disp_tensor = torch.zeros((1, self.height, self.width), dtype=torch.float32)

        return {
            "image": image, 
            "disparity": disp_tensor,
            "has_depth": has_depth,
            "prompt": self.prompt
        }

# -------------------------
# Training Module
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

    def set_accelerator(self, accelerator):
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
            "raw_disparity": data["disparity"],
            "has_depth": data["has_depth"]
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

        # 1. Prepare Disparity Map
        disparity_map = inputs["raw_disparity"].to(self.pipe.device).float()
        
        # Ensure 4D shape (N, 1, H, W)
        if disparity_map.ndim == 3:
            disparity_map = disparity_map.unsqueeze(0)

        has_depth = inputs["has_depth"]
        target_disparity = None

        if has_depth:
            # === Sky Mask Handling ===
            # Identify sky pixels (very low disparity < 0.01)
            sky_mask = (disparity_map < 0.01).bool()
            
            # Replace sky with mean of valid pixels to stabilize noise statistics
            if (~sky_mask).any():
                disparity_map[sky_mask] = disparity_map[~sky_mask].mean()
            
            target_disparity = disparity_map

        input_latents = inputs["input_latents"]
        h, w = input_latents.shape[-2:]
        
        # 2. Sample Radius Parameters
        cutoff_radius = min(np.random.exponential(scale=1/0.1), min(h, w) // 2)
        
        if has_depth:
            # Mode A: Depth-Guided (Cutoff < Maximal)
            factor = np.random.uniform(1.0, 4.0)
            maximal_radius = min(cutoff_radius * factor, min(h, w) // 2)
        else:
            # Mode B: Standard Cutoff (Cutoff == Maximal)
            maximal_radius = cutoff_radius

        gamma = float(np.random.uniform(1.0, 10.0))
        
        input_noise = torch.randn_like(input_latents.float())
        
        # 3. Generate Noise
        structured_noise = generate_wavelet_structured_noise_batch_vectorized(
            input_latents.float(), 
            cutoff_radius=cutoff_radius, 
            maximal_radius=maximal_radius,
            disparity_map=target_disparity,
            gamma=gamma, 
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
                        "noise/cutoff_radius": float(cutoff_radius),
                        "noise/maximal_radius": float(maximal_radius),
                        "noise/gamma": float(gamma),
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
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to local OpenScene dataset")
    parser.add_argument("--max_samples", type=int, default=None, help="Debug: limit samples")
    args = parser.parse_args()

    dataset = OpenSceneDataset(
        root_dir=args.dataset_path,
        max_samples=args.max_samples,
        height=args.height,
        width=args.width,
    )

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