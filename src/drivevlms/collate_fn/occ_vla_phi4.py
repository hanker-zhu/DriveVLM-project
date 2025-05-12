from transformers import BatchFeature
from ..registry import register_collate_fn
from PIL import Image

def format_prompt_phi4(instruction, input=None):

    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "<|user|><|image_1|><|image_2|><|image_3|><|image_4|><|image_5|><|image_6|>"
            "Below is an instruction describing a driving perception task, along with six images from different views around the ego vehicle.\n"
            "Each image corresponds to a specific camera: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT.\n"
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:<|end|><|assistant|>"
        )
    }
    if input is None:
        return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
    else:
        return PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})
    
def format_answer(answer):
    return answer + '<|end|><|endoftext|>'

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def cat_with_pad(tensors, dim, padding_value=0):
    """
    cat along dim, while pad to max for all other dims
    """
    ndim = tensors[0].dim()
    assert all(
        t.dim() == ndim for t in tensors[1:]
    ), 'All tensors must have the same number of dimensions'

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        # Create a slice list where every dimension except dim is full slice
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        # Update only the concat dimension slice
        slices[dim] = slice(index, index + t.shape[dim])

        output[slices] = t
        index += t.shape[dim]

    return output

import torch
import copy
_IGNORE_INDEX = -100
_MAX_TRAINING_LENGTH = 8192
@register_collate_fn
def occ_vla_phi4_collate_fn_train(examples, processor, dtype):
    prompts = ["<|user|><|image_1|><|image_2|><|image_3|><|image_4|><|image_5|><|image_6|>answer en " \
                + example["input"] + "<|end|><|assistant|>" for example in examples]
    answers =  [example["output"] + "<|end|><|endoftext|>" for example in examples]
    print(prompts)
    print(answers)
    images = []
    for example in examples:
        camera_views = [
            example[cam]
            for cam in [
                "cam_front",
                "cam_front_right",
                "cam_front_left",
                "cam_back",
                "cam_back_left",
                "cam_back_right",
            ]
        ]
        camera_views = [cam.replace('/root/shared/', '/data2/public-data/') for cam in camera_views]
        images.append([Image.open(cam).convert("RGB") for cam in camera_views])


    input_ids_list = []
    labels_list = []
    input_image_embeds_list = []
    image_attention_mask_list = []
    image_sizes_list = []

    for prompt, answer, image in zip(prompts, answers, images):
        image = [img.resize((448, 448), ) for img in image]
        inputs = processor([prompt], images=image, return_tensors='pt')
        answer_ids = processor.tokenizer(answer, return_tensors='pt').input_ids
        input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
        labels = copy.deepcopy(input_ids)
        labels[:, -answer_ids.shape[1] :] = answer_ids
        labels[:, -answer_ids.shape[1] :] = answer_ids

        if input_ids.size(1) > _MAX_TRAINING_LENGTH:
            input_ids = input_ids[:, :_MAX_TRAINING_LENGTH]
            labels = labels[:, :_MAX_TRAINING_LENGTH]
            if torch.all(labels == _IGNORE_INDEX).item():
                # workaround to make sure loss compute won't fail
                labels[:, -1] = processor.tokenizer.eos_token_id
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        input_image_embeds_list.append(inputs.input_image_embeds)
        image_attention_mask_list.append(inputs.image_attention_mask)
        image_sizes_list.append(inputs.image_sizes)

    input_ids = pad_sequence(input_ids_list, padding_side='right', padding_value=0)
    labels = pad_sequence(labels_list, padding_side='right', padding_value=0)
    attention_mask = (input_ids != 0).long()
    input_image_embeds = cat_with_pad(input_image_embeds_list, dim=0)
    image_attention_mask = cat_with_pad(image_attention_mask_list, dim=0)
    image_sizes = torch.cat(image_sizes_list)
    return BatchFeature(
        {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'input_image_embeds': input_image_embeds,
            'image_attention_mask': image_attention_mask,
            'image_sizes': image_sizes,
            'input_mode': 1,  # vision mode
        }
    )

@register_collate_fn
def occ_vla_phi4_collate_fn_val(examples, processor, device):
    ids = [example["id"] for example in examples]
    questions = [example["conversations"][0]['value'] for example in examples]
    prompts = [format_prompt_phi4(example["conversations"][0]['value']) for example in examples]
    images = []
    for example in examples:
        image = [Image.open(example["image_paths"][i]).convert("RGB") for i in range(6)]
        images.append(image)
    # Currently only support batchsize = 1
    image = [img.resize((448, 448), ) for img in images[0]]
    tokens = processor(
        text=prompts, images=image, return_tensors="pt", padding="longest"
    )

    return tokens, questions, ids


