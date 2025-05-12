import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from transformers import AutoModelForImageTextToText
from datasets import load_from_disk
from PIL import Image
import json
from tqdm import tqdm
import argparse
import re
import os
from torchvision import transforms
from peft import PeftModel

_imgs_filename = [
            "/data2/public-data/DriveLM-nuScenes/val_data/CAM_FRONT/n008-2018-09-18-14-35-12-0400__CAM_FRONT__1537295990612404.jpg",
            "/data2/public-data/DriveLM-nuScenes/val_data/CAM_FRONT_LEFT/n008-2018-09-18-14-35-12-0400__CAM_FRONT_LEFT__1537295990604799.jpg",
            "/data2/public-data/DriveLM-nuScenes/val_data/CAM_FRONT_RIGHT/n008-2018-09-18-14-35-12-0400__CAM_FRONT_RIGHT__1537295990620482.jpg",
            "/data2/public-data/DriveLM-nuScenes/val_data/CAM_BACK/n008-2018-09-18-14-35-12-0400__CAM_BACK__1537295990637558.jpg",
            "/data2/public-data/DriveLM-nuScenes/val_data/CAM_BACK_LEFT/n008-2018-09-18-14-35-12-0400__CAM_BACK_LEFT__1537295990647405.jpg",
            "/data2/public-data/DriveLM-nuScenes/val_data/CAM_BACK_RIGHT/n008-2018-09-18-14-35-12-0400__CAM_BACK_RIGHT__1537295990628113.jpg"
        ]

processor = AutoProcessor.from_pretrained('google/paligemma2-3b-pt-224')
model = PaliGemmaForConditionalGeneration.from_pretrained(
    'google/paligemma2-3b-pt-224',
    torch_dtype=torch.float16,
)

# model = PeftModel.from_pretrained(model, '/data2/private-data/zhangn/pretrained/paligemma/FULL-2025-03-15_21-49/final_model/')
# model = model.merge_and_unload()

model.to('cuda')

import types

# 假设 model 是你的模型实例
original_forward = model.forward

def patched_forward(self, *args, last_cache_position=None, **kwargs):
    # 如果 kwargs 中存在 last_cache_position，就将其移除（或根据需要做处理）
    if "last_cache_position" in kwargs:
        kwargs.pop("last_cache_position")
    return original_forward(*args, **kwargs)

# 用新的方法替换旧的 forward
model.forward = types.MethodType(patched_forward, model)


def infer(inputs):

    input_len = inputs["input_ids"].shape[-1]
    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        num_beams=1,
        do_sample=True,
        top_p=0.8,
        temperature=0.5,  
        repetition_penalty=1.02
    )
    output = output[:, input_len:]

    results = processor.batch_decode(output, skip_special_tokens=True)
    return results


def format_prompt(instruction, input=None):

    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    }
    if input is None:
        return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
    else:
        return PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})


def tokenize(texts, images, processor, device='cuda'):
    return processor(
        text=texts, images=images, return_tensors="pt", padding="longest"
    ).to(device)


@torch.no_grad()
def main(args):

    sample_prompts = [
        "What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision.",
        "What is the moving status of object <c1,CAM_BACK,384.2,477.5>? Please select the correct answer from the following options: A. Turn right. B. Drive backward. C. Going ahead. D. Turn left."
    ]


    for instruction in sample_prompts:
        prompt = format_prompt(instruction)
        print(f"\nGenerating for prompt: {repr(prompt)}")
        images = [Image.open(cam).convert("RGB") for cam in _imgs_filename]

        reason_inputs = tokenize([prompt], [images], processor, args.device)
        reason_results = infer(reason_inputs)
        print(reason_results)

def parse_args():
    parser = argparse.ArgumentParser(description='DriveLM PaliGemma2 Inference Demo')
    parser.add_argument("--device", default="cuda", help="Device to run inference")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_args())