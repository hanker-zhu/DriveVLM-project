import torch
from typing import Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class DriveLMNusPhi4Config:
    model_name: str = "microsoft/Phi-4-multimodal-instruct"
    model_preparation: str = "prepare_model_and_processor_phi4"
    collate_fn_train: str = "drivelm_nus_phi4_collate_fn"
    collate_fn_val: str = None
    peft_name: Optional[str] = None
    dataset_name: str = "data/DriveLM_nuScenes/train"
    wandb_project = None
    run_name: str = f"FULL-{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    output_dir: str = "/data2/private-data/zhangn/pretrained/phi4/" + f"{run_name}"

    num_train_epochs: int = 3
    batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 8
    lr: float = 5e-6
    lora_r: int = 32
    warmup_steps: int = 500
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0

    seed: int = 42
    dtype = torch.bfloat16
    quantization: bool = False
    use_flash_attention: bool = False
    use_lora: bool = True

    resume_from_checkpoint: bool = False
    save_lora_adapter_when_checkpointing: bool = False

    save_steps: int = 500
    log_steps: int = 10  # log to wandb & tensorboard, gathered loss (slower)
    print_steps: int = 10  # local print, loss on GPU0

    find_unused_parameters: bool = True

config = DriveLMNusPhi4Config()