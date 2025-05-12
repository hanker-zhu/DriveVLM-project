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

# processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
# model = AutoModelForImageTextToText.from_pretrained("google/paligemma-3b-pt-224")
processor = AutoProcessor.from_pretrained("lykong/paligemma-finetuned")
model = AutoModelForImageTextToText.from_pretrained("lykong/paligemma-finetuned")
# model = PeftModel.from_pretrained(model, '/data2/private-data/zhangn/pretrained/paligemma/FULL-2025-04-11_11-39/epoch-1')
# model = PeftModel.from_pretrained(model, '/data2/private-data/zhangn/pretrained/paligemma/FULL-2025-03-15_21-49/final_model')
# model = model.merge_and_unload()
model.to('cuda')

import cv2
import os
import numpy as np

def visualize(imgs_filename, detection_text, save_path="visualized_output.jpg"):
    # 如果 detection_text 是列表，则取第一个字符串
    if isinstance(detection_text, list):
        detection_text = detection_text[0]
    # 使用正则表达式提取所有尖括号内的内容
    matches = re.findall(r"<([^>]+)>", detection_text)
    detections = []
    for match in matches:
        parts = match.split(",")
        if len(parts) == 4:
            obj_id, cam, x, y = parts
            detections.append({
                "id": obj_id.strip(),
                "camera": cam.strip(),
                "x": float(x),
                "y": float(y)
            })

    # 将6张图片读入字典中（key 是相机名称 CAM_FRONT 等）
    images = {}
    for img_path in imgs_filename:
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Warning: Failed to read {img_path}")
            continue
        # 获取相机名，如 CAM_FRONT
        cam = os.path.basename(img_path).split("__")[1]
        images[cam] = img

    # 在图上绘制检测结果
    for det in detections:
        cam = det["camera"]
        x, y = int(det["x"]), int(det["y"])
        obj_id = det["id"]

        if cam in images:
            img = images[cam]
            cv2.circle(img, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(img, obj_id, (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2)

    # 定义前置与后置摄像头的顺序
    front_order = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
    back_order  = ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
    target_size = (640, 360)  # (宽, 高)

    def get_resized_image(images):
        return cv2.resize(images, target_size)

    # 分别用列表推导获取每一排的缩放图片
    row_front = cv2.hconcat([get_resized_image(images[cam]) for cam in front_order])
    row_back  = cv2.hconcat([get_resized_image(images[cam]) for cam in back_order])

    # 纵向拼接两排
    combined_image = cv2.vconcat([row_front, row_back])

    # 保存图片
    cv2.imwrite(save_path, combined_image)
    print(f"✅ Visualized image saved to: {save_path}")


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
        # "prompt_no_input": (
        #     "Below is an instruction that describes a task. "
        #     "Write a response that appropriately completes the request.\n\n"
        #     "### Instruction:\n{instruction}\n\n### Response:"
        # ),
        "prompt_no_input": (
            "Based on image inputs from the vehicle's six surround-view cameras(CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT,CAM_BACK,CAM_BACK_LEFT,CAM_BACK_RIGHT). "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        # ),
        "prompt_no_input": (
            "answer en You are an   driving labeler. You have access to six camera images (front<image_00>, front-right<image>, front-left<image>, back<image>, back-right<image>, back-left<image>)"
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
        # "What is the moving status of object <c1,CAM_BACK,384.2,477.5>? Please select the correct answer from the following options: A. Turn right. B. Drive backward. C. Going ahead. D. Turn left."
    ]

    for instruction in sample_prompts:
        prompt = format_prompt(instruction)
        print(f"\nGenerating for prompt: {repr(prompt)}")
        images = [Image.open(cam).convert("RGB") for cam in _imgs_filename]

        reason_inputs = tokenize([prompt], images, processor, args.device)
        reason_results = infer(reason_inputs)
        print(reason_results)

        visualize(_imgs_filename, reason_results)

def parse_args():
    parser = argparse.ArgumentParser(description='DriveLM PaliGemma Inference Demo')
    parser.add_argument("--device", default="cuda", help="Device to run inference")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_args())