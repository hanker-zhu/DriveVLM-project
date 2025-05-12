from PIL import Image
from ..registry import register_collate_fn

def format_prompt(instruction, input=None):

    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "You are an autonomous driving labeler. You have access to six camera images (front, front-right, front-left, back, back-right, back-left).\n{instruction}"
        ),
    }
    if input is None:
        return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
    else:
        return PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})
    
@register_collate_fn
def drivelm_nus_paligemma_collate_fn_train(examples, processor, dtype):
    assert len(examples) == 1
    texts = [format_prompt(example["conversations"][0]['value']) for example in examples]
    labels = [example["conversations"][1]['value'] for example in examples]
    images = []
    for example in examples:
        image = [Image.open(example["image_paths"][i]).convert("RGB") for i in range(6)]
        images.append(image)
    tokens = processor(
        text=texts, images=images, suffix=labels, return_tensors="pt", padding="longest"
    )
    return tokens.to(dtype)

    
@register_collate_fn
def drivelm_nus_paligemma_collate_fn_val(examples, processor, dtype):
    assert len(examples) == 1

    ids = [example["id"] for example in examples]
    questions = [example["conversations"][0]['value'] for example in examples]
    prompts = [format_prompt(example["conversations"][0]['value']) for example in examples]
    images = []
    for example in examples:
        image = [Image.open(example["image_paths"][i]).convert("RGB") for i in range(6)]
        images.append(image)

    tokens = processor(
        text=prompts, images=images, return_tensors="pt", padding="longest"
    )
    return tokens, questions, ids

