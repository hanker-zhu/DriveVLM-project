import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers import GenerationConfig
from PIL import Image
import argparse
import os



_imgs_filename = ["/data2/public-data/DriveLM-nuScenes/val_data/CAM_FRONT/n008-2018-08-30-10-33-52-0400__CAM_FRONT__1535639711662404.jpg",
                    "/data2/public-data/DriveLM-nuScenes/val_data/CAM_FRONT_LEFT/n008-2018-08-30-10-33-52-0400__CAM_FRONT_LEFT__1535639711654799.jpg",
                    "/data2/public-data/DriveLM-nuScenes/val_data/CAM_FRONT_RIGHT/n008-2018-08-30-10-33-52-0400__CAM_FRONT_RIGHT__1535639711670482.jpg",
                    "/data2/public-data/DriveLM-nuScenes/val_data/CAM_BACK/n008-2018-08-30-10-33-52-0400__CAM_BACK__1535639711687558.jpg",
                    "/data2/public-data/DriveLM-nuScenes/val_data/CAM_BACK_LEFT/n008-2018-08-30-10-33-52-0400__CAM_BACK_LEFT__1535639711697405.jpg",
                    "/data2/public-data/DriveLM-nuScenes/val_data/CAM_BACK_RIGHT/n008-2018-08-30-10-33-52-0400__CAM_BACK_RIGHT__1535639711678113.jpg"]

# Load model and processor
processor = AutoProcessor.from_pretrained("microsoft/Phi-4-multimodal-instruct",  revision="607bf62a754018e31fb4b55abbc7d72cce4ffee5",
                                          trust_remote_code=True)


# 官网模型 
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-multimodal-instruct", 
    torch_dtype=torch.float16, 
    _attn_implementation='sdpa',
    revision="607bf62a754018e31fb4b55abbc7d72cce4ffee5",
    trust_remote_code=True
)

# model = AutoModelForCausalLM.from_pretrained(
#     '/data2/private-data/zhangn/pretrained/phi4/FULL-2025-03-24_21-59/hf_ckpt',
#     torch_dtype=torch.float16, 
#     _attn_implementation='sdpa',
#     revision ="607bf62a754018e31fb4b55abbc7d72cce4ffee5",
#     trust_remote_code=True
# )

#TODO since phi4 finetune vision lora is self-contained in model
# the loading method shoud be differenct maybe

# model = PeftModel.from_pretrained(model, '/data2/private-data/zhangn/pretrained/paligemma/FULL-2025-03-15_21-49/final_model/')
# model = model.merge_and_unload()

model.to('cuda')

# Load generation config
generation_config = GenerationConfig.from_pretrained("microsoft/Phi-4-multimodal-instruct", revision="607bf62a754018e31fb4b55abbc7d72cce4ffee5")

# Define prompt structure
user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

def infer(inputs):

    input_len = inputs["input_ids"].shape[-1]
    output = model.generate(
        **inputs,
        max_new_tokens=1000,
        generation_config=generation_config
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
        #     "<|user|><|image_1|><|image_2|><|image_3|><|image_4|><|image_5|><|image_6|>Below is an instruction that describes a task. "
        #     "Write a response that appropriately completes the request.\n\n"
        #     "### Instruction:\n{instruction}\n\n### Response:<|end|><|assistant|>"
        # ),
        "prompt_no_input": 
            ("<|user|><|image_1|><|image_2|><|image_3|><|image_4|><|image_5|><|image_6|>You are an Autonomous Driving AI assistant. You receive an image that consists of six surrounding camera views. These six images are the front view, front left view, front right view, back view, back left view and back right view of the ego vehicle. Your task is to analyze these images and provide insights or actions based on the visual data." + instruction + "<|end|><|assistant|>")
        # "prompt_no_input": (
        #     "<|user|><|image_1|><|image_2|><|image_3|><|image_4|><|image_5|><|image_6|>"
        #     "Below is an instruction describing a driving perception task, along with six images from different views around the ego vehicle.\n"
        #     "Each image corresponds to a specific camera: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT.\n"
        #     "If the instruction refers to important objects in the current scene, please identify all visible objects and annotate them with:\n"
        #     "- Object ID (e.g., <c1>)\n"
        #     "- Camera name (e.g., CAM_BACK)\n"
        #     "- Pixel coordinates of the bounding box center (x, y)\n"
        #     "- Object type (car, truck, pedestrian, etc.)\n\n"
        #     "Important:\n"
        #     "- Be accurate with pixel coordinates (in the format: <id, camera, x_pixel, y_pixel>)\n"
        #     "- Only include objects visible in the respective image\n"
        #     "- Do not invent objects not present\n"
        #     "- Each object should be described once per camera\n\n"
        #     "Otherwise, feel free to provide a natural language response.\n\n "
        #     "### Instruction:\n{instruction}\n\n### Response:<|end|><|assistant|>"
        # )
        # TODO currently format_prompts support 6 images
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
        'What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision.'
    ]

    for instruction in sample_prompts:
        prompt = format_prompt(instruction)
        print(f"\nGenerating for prompt: {repr(prompt)}")
        images = [Image.open(cam).convert("RGB") for cam in _imgs_filename]
        images = [image.resize((448, 448), ) for image in images]
        reason_inputs = tokenize([prompt], images, processor, args.device)
        reason_results = infer(reason_inputs)
        print(reason_results)

def parse_args():
    parser = argparse.ArgumentParser(description='DriveLM Phi4 Inference')
    parser.add_argument("--device", default="cuda", help="Device to run inference")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_args())