import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import glob
import random
import cv2  # Added for robust 16-bit depth loading
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from datetime import datetime

# Import WanVideo components
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, wan_parser
from examples.wanvideo.model_training.train import WanTrainingModule
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration

# Import your PPD Noise Generator
from wavelet_noise import generate_wavelet_structured_noise_batch_vectorized

# -----------------------------------------------------------------------------
# 1. OpenScene Video Dataset
# -----------------------------------------------------------------------------
class OpenSceneVideoDataset(Dataset):
    def __init__(self, root_dir, height=704, width=1280, 
                 target_frames=[9, 17, 33, 49], 
                 prompt="A photorealistic driving scene in a city, view from a car dashboard. Natural lighting, urban buildings, trees, cars on the street. High resolution, realistic textures.",
                 max_samples=None):
        self.root_dir = root_dir
        self.height = height
        self.width = width
        self.target_frames = sorted(target_frames)
        self.prompt = prompt
        
        print(f"Scanning dataset at {root_dir}...")
        
        self.labeled_clips = []
        self.unlabeled_clips = []
        
        # 1. Scan for clips
        # Structure: root_dir / log_id / clip_id / images
        clip_image_folders = sorted(glob.glob(os.path.join(root_dir, "*", "*", "images")))
        
        print(f"Found {len(clip_image_folders)} potential clips. Checking validity...")
        
        for img_dir in tqdm(clip_image_folders, desc="Indexing"):
            clip_root = os.path.dirname(img_dir)
            disp_dir = os.path.join(clip_root, "disparity")
            
            # Check for _DONE flag to confirm depth is valid
            is_labeled = os.path.exists(os.path.join(disp_dir, "_DONE"))
            
            # Count frames
            # Assuming standard naming, we can listdir or glob
            # To be faster, we might rely on the folder existing, but let's check one frame count
            # Optimization: only count if we need to.
            frames = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
            count = len(frames)
            
            if count >= self.target_frames[0]:
                clip_info = {
                    "img_dir": img_dir,
                    "depth_dir": disp_dir if is_labeled else None,
                    "frames": frames,
                    # If labeled, we assume depths match frame names (.jpg -> .png)
                    "supported_lengths": [L for L in self.target_frames if count >= L]
                }
                
                if is_labeled:
                    self.labeled_clips.append(clip_info)
                else:
                    self.unlabeled_clips.append(clip_info)
        
        print(f"Dataset Summary:")
        print(f"  - Labeled Clips (Images + Valid Depth): {len(self.labeled_clips)}")
        print(f"  - Unlabeled Clips (Images Only): {len(self.unlabeled_clips)}")
        
        self.virtual_length = max(len(self.labeled_clips), len(self.unlabeled_clips))
        if max_samples:
            self.virtual_length = min(self.virtual_length, max_samples)

    def __len__(self):
        return self.virtual_length

    def __getitem__(self, idx):
        # 50/50 Balanced Sampling
        use_labeled = random.random() < 0.5
        if len(self.labeled_clips) == 0: use_labeled = False
        if len(self.unlabeled_clips) == 0: use_labeled = True
        
        if use_labeled:
            clip = random.choice(self.labeled_clips)
            has_depth = True
        else:
            clip = random.choice(self.unlabeled_clips)
            has_depth = False
            
        # Select Random Length & Start
        target_len = random.choice(clip["supported_lengths"])
        start_idx = random.randint(0, len(clip["frames"]) - target_len)
        
        sel_frames = clip["frames"][start_idx : start_idx + target_len]
        
        video = []
        disparity_tensors = []
        
        for i, f in enumerate(sel_frames):
            # 1. Load Image
            img_path = os.path.join(clip["img_dir"], f)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.width, self.height), Image.LANCZOS)
            video.append(img)
            
            # 2. Load Depth (if exists)
            if has_depth:
                png_name = f.replace(".jpg", ".png")
                d_path = os.path.join(clip["depth_dir"], png_name)
                
                # Use cv2 for reliable 16-bit loading as per reference
                disp_16bit = cv2.imread(d_path, cv2.IMREAD_UNCHANGED)
                if disp_16bit is None:
                    # Fallback to zero if read fails
                    disp_tensor = torch.zeros((1, self.height, self.width), dtype=torch.float32)
                else:
                    disp_16bit = cv2.resize(disp_16bit, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                    # Normalize 0~65535 -> 0.0~1.0
                    disp_float = disp_16bit.astype(np.float32) / 65535.0
                    disp_tensor = torch.from_numpy(disp_float).unsqueeze(0)
            else:
                disp_tensor = torch.zeros((1, self.height, self.width), dtype=torch.float32)
            
            disparity_tensors.append(disp_tensor)
    
        disparity_tensors = torch.stack(disparity_tensors) # (T, 1, H, W)
        
        return {
            "video": video,
            "disparity": disparity_tensors, 
            "has_depth": has_depth,
            "prompt": self.prompt
        }

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None, audio_processor_config=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        if audio_processor_config is not None:
            audio_processor_config = ModelConfig(model_id=audio_processor_config.split(":")[0], origin_file_pattern=audio_processor_config.split(":")[1])
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, audio_processor_config=audio_processor_config)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
        # Accelerator will be injected after accelerator.prepare(...)
        self.accelerator: Accelerator | None = None
       
    def set_accelerator(self, accelerator):
        self.accelerator = accelerator
        self.pipe.device = accelerator.device
        self.pipe.vae_device = accelerator.device
        self.pipe.enable_cpu_offload()
        if hasattr(self.pipe, "color_embed"):
            self.pipe.color_embed.to(self.pipe.device)

    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": True,
            "tile_size": (30, 52),
            "tile_stride": (15, 26),
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "raw_disparity": data["disparity"],
            "has_depth": data["has_depth"],
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None, step: int | None = None):
        if inputs is None: inputs = self.forward_preprocess(data)
        
        # Load models back to GPU for training
        self.pipe.load_models_to_device(self.pipe.in_iteration_models)
        
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        
        input_latents = inputs["input_latents"]
        _, _, t, h, w = input_latents.shape
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
            tmp = target_disparity.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 1, T, H, W)
            tmp = F.interpolate(tmp, size=(t, h, w), mode="trilinear", align_corners=False)
            target_disparity = tmp.squeeze(0).permute(1, 0, 2, 3)  # (t, 1, h, w)

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
        
        input_noise = torch.randn_like(input_latents[0].transpose(0, 1).float())
        
        # 3. Generate Noise
        structured_noise = generate_wavelet_structured_noise_batch_vectorized(
            input_latents[0].transpose(0, 1).float(), 
            cutoff_radius=cutoff_radius, 
            maximal_radius=maximal_radius,
            disparity_map=target_disparity,
            gamma=gamma, 
            input_noise=input_noise,
        )
        structured_noise = structured_noise.transpose(0, 1)[None].contiguous()

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

# -----------------------------------------------------------------------------
# 3. Main Launcher
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    dataset = OpenSceneVideoDataset(
        root_dir=args.dataset_base_path,
        height=args.height, width=args.width,
        target_frames=[args.num_frames]
    )

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=args.audio_processor_config,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )

    model_logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    launch_training_task(dataset, model, model_logger, args=args)