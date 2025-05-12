from PIL import Image
from ..registry import register_collate_fn

@register_collate_fn
def occ_vla_paligemma_collate_fn_train(examples, processor, dtype):
    """
    Args:
        examples: List[Dict], the lenght of list is the batchsize.
    """
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
    texts = None
    labels = None
    tokens = processor(
        text=texts, images=images, suffix=labels, return_tensors="pt", padding="longest"
    )
    return tokens.to(dtype)


@register_collate_fn
def occ_vla_paligemma_collate_fn_val(examples, processor, dtype):
    history = [example["history"] for example in examples]
    future = [example["future"] for example in examples]
    desc = [example["desc"] for example in examples]
    reason = [example["reason"] for example in examples]
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
    return history, future, desc, reason, images