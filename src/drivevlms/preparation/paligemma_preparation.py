import torch
from accelerate import PartialState
from transformers import (
    PaliGemmaProcessor,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model, PeftModel
from ..registry import register_prepare_model_and_processor

@register_prepare_model_and_processor
def prepare_model_and_processor_paligemma(config):
    processor = PaliGemmaProcessor.from_pretrained(config.model_name)

    model = None
    if config.quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=config.dtype,
        )
        if config.use_flash_attention:
            assert (
                config.dtype == torch.bfloat16
            ), "Flash attention only supports bfloat16"
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                config.model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": PartialState().local_process_index},
            )
        else:
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                config.model_name, quantization_config=bnb_config
            )
    else:
        if config.use_flash_attention:
            assert (
                config.dtype == torch.bfloat16
            ), "Flash attention only supports bfloat16"
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                config.model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": PartialState().local_process_index},
            )
        else:
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                    config.model_name, torch_dtype=torch.bfloat16,)

    if config.peft_name:
        model = PeftModel.from_pretrained(model, config.peft_name)
        model = model.merge_and_unload()

    # finetune vision encoder
    for param in model.vision_tower.parameters():
        param.requires_grad = True

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = True

    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            target_modules=[
                "q_proj",
                "o_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    else:
        for param in model.language_model.parameters():
            param.requires_grad = True
    return model, processor