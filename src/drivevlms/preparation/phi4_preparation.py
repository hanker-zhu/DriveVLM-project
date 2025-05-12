
import torch
from accelerate import PartialState
from transformers import (AutoProcessor, 
                        AutoModelForCausalLM, 
                        BitsAndBytesConfig)
from ..registry import register_prepare_model_and_processor
import copy
from peft import LoraConfig
from peft.tuners.lora.layer import LoraLayer
from drivevlms.models.phi4_bjxx import Phi4MMProcessor, Phi4MMForCausalLM
from accelerate import Accelerator

@register_prepare_model_and_processor
def prepare_model_and_processor_phi4(config):
    processor = AutoProcessor.from_pretrained(config.model_name, 
                                              trust_remote_code=True,
                                              revision="607bf62a754018e31fb4b55abbc7d72cce4ffee5")
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
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": PartialState().local_process_index},
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name, 
                quantization_config=bnb_config
            )
    else:
        if config.use_flash_attention:
            assert (
                config.dtype == torch.bfloat16
            ), "Flash attention only supports bfloat16"
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": PartialState().local_process_index},
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                    config.model_name, 
                    _attn_implementation="sdpa", 
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    use_cache=False,
                    revision="607bf62a754018e31fb4b55abbc7d72cce4ffee5")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    # remove parameters irrelevant to vision tasks
    del model.model.embed_tokens_extend.audio_embed  # remove audio encoder
    for layer in model.model.layers:
        # remove audio lora
        del layer.mlp.down_proj.lora_dropout.speech
        del layer.mlp.down_proj.lora_A.speech
        del layer.mlp.down_proj.lora_B.speech
        del layer.mlp.gate_up_proj.lora_dropout.speech
        del layer.mlp.gate_up_proj.lora_A.speech
        del layer.mlp.gate_up_proj.lora_B.speech
        del layer.self_attn.o_proj.lora_dropout.speech
        del layer.self_attn.o_proj.lora_A.speech
        del layer.self_attn.o_proj.lora_B.speech
        del layer.self_attn.qkv_proj.lora_dropout.speech
        del layer.self_attn.qkv_proj.lora_A.speech
        del layer.self_attn.qkv_proj.lora_B.speech


    # tune vision encoder and lora
    model.set_lora_adapter('vision')
    for param in model.model.embed_tokens_extend.image_embed.parameters():
        param.requires_grad = True
    return model, processor


@register_prepare_model_and_processor
def prepare_model_and_processor_phi4_add_lora(config):
    processor = AutoProcessor.from_pretrained(config.model_name, 
                                              trust_remote_code=True,
                                              revision="607bf62a754018e31fb4b55abbc7d72cce4ffee5")

    model = Phi4MMForCausalLM.from_pretrained(
            config.model_name, 
            _attn_implementation="sdpa", 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            revision="607bf62a754018e31fb4b55abbc7d72cce4ffee5")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    # remove parameters irrelevant to vision tasks
    del model.model.embed_tokens_extend.audio_embed  # remove audio encoder
    for layer in model.model.layers:
        # remove audio lora
        del layer.mlp.down_proj.lora_dropout.speech
        del layer.mlp.down_proj.lora_A.speech
        del layer.mlp.down_proj.lora_B.speech
        del layer.mlp.gate_up_proj.lora_dropout.speech
        del layer.mlp.gate_up_proj.lora_A.speech
        del layer.mlp.gate_up_proj.lora_B.speech
        del layer.self_attn.o_proj.lora_dropout.speech
        del layer.self_attn.o_proj.lora_A.speech
        del layer.self_attn.o_proj.lora_B.speech
        del layer.self_attn.qkv_proj.lora_dropout.speech
        del layer.self_attn.qkv_proj.lora_A.speech
        del layer.self_attn.qkv_proj.lora_B.speech

    # 激活domain
    model.set_lora_adapter('domain')
    return model, processor



@register_prepare_model_and_processor
def prepare_model_and_processor_phi4_merge_vision(config):
    processor = AutoProcessor.from_pretrained(config.model_name, 
                                              trust_remote_code=True,
                                              revision="607bf62a754018e31fb4b55abbc7d72cce4ffee5")

    model = Phi4MMForCausalLM.from_pretrained(
            config.model_name, 
            _attn_implementation="sdpa", 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False)

    # remove parameters irrelevant to vision tasks
    del model.model.embed_tokens_extend.audio_embed  # remove audio encoder
    for layer in model.model.layers:
        # remove audio lora
        del layer.mlp.down_proj.lora_dropout.speech
        del layer.mlp.down_proj.lora_A.speech
        del layer.mlp.down_proj.lora_B.speech
        del layer.mlp.gate_up_proj.lora_dropout.speech
        del layer.mlp.gate_up_proj.lora_A.speech
        del layer.mlp.gate_up_proj.lora_B.speech
        del layer.self_attn.o_proj.lora_dropout.speech
        del layer.self_attn.o_proj.lora_A.speech
        del layer.self_attn.o_proj.lora_B.speech
        del layer.self_attn.qkv_proj.lora_dropout.speech
        del layer.self_attn.qkv_proj.lora_A.speech
        del layer.self_attn.qkv_proj.lora_B.speech

    model.set_lora_adapter('vision')
    # return model, processor
    def merge_and_remove_lora(model):
        """
        合并 vision LoRA 权重到 base_layer，并删除 vision 和 speech LoRA。
        """
        vision_lora_layers = []
        speech_lora_layers = []

        # 查找 vision 和 speech LoRA 层
        for name, module in list(model.named_modules()):
            if isinstance(module, LoraLayer):
                if "vision" in module.lora_A.keys():
                    vision_lora_layers.append((name, module))
                elif "speech" in module.lora_A.keys():
                    speech_lora_layers.append((name, module))
        # 合并 vision LoRA 权重
        for name, module in vision_lora_layers:
            if module.merged:
                warnings.warn(f"Layer {name} is already merged, skipping.")
            else:
                try:
                    module.merge()
                    #将lora层替换为合并后的base_layer
                    parent_name = '.'.join(name.split('.')[:-1]) # 'model.layers.0.self_attn'
                    parent_module = model.get_submodule(parent_name)
                    setattr(parent_module, name.split('.')[-1], module.base_layer)
                except Exception as e:
                    warnings.warn(f"Merge layer {name} failed, error: {e}")

    merge_and_remove_lora(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model, processor