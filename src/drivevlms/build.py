from .registry import (COLLATE_FN_REGISTRY,
                       PREPARE_REGISTRY)

def build_collate_fn(name):
    if name not in COLLATE_FN_REGISTRY:
        raise ValueError(
            f"Collate function {name} not found. "
            f"Available functions: {list(COLLATE_FN_REGISTRY.keys())}"
        )
    return COLLATE_FN_REGISTRY[name]

def build_preparation(name):
    if name not in PREPARE_REGISTRY:
        raise ValueError(
            f"Collate function {name} not found. "
            f"Available functions: {list(COLLATE_FN_REGISTRY.keys())}"
        )
    return PREPARE_REGISTRY[name]