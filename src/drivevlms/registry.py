COLLATE_FN_REGISTRY = {}
PREPARE_REGISTRY = {}

def register_collate_fn(fn):
    COLLATE_FN_REGISTRY[fn.__name__] = fn
    return fn

def register_prepare_model_and_processor(fn):
    PREPARE_REGISTRY[fn.__name__] = fn
    return fn