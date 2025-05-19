import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig
# from peft import PeftModel # PeftModel might be needed if loading LoRA adapters before pruning
import argparse
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
from functools import partial
import json
import time
import torch.nn.utils.prune as prune
from drivevlms.build import build_collate_fn
import numpy as np

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig
from peft import PeftModel
import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
from functools import partial
import argparse
from drivevlms.build import build_collate_fn
import json


def apply_pruning(model, prune_llm_amount, prune_vision_amount):
    """
    Apply L1 unstructured pruning to the model.
    Note: This is a basic demonstration. For production, prune then fine-tune.
    """
    if prune_llm_amount > 0 and hasattr(model, 'language_model'):
        print(f"Applying L1 unstructured pruning to LLM with amount: {prune_llm_amount}")
        parameters_to_prune_llm = []
        for module_name, module in model.language_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                parameters_to_prune_llm.append((module, 'weight'))
        
        if parameters_to_prune_llm:
            prune.global_unstructured(
                parameters_to_prune_llm,
                pruning_method=prune.L1Unstructured,
                amount=prune_llm_amount,
            )
            # Make pruning permanent
            for module, name in parameters_to_prune_llm:
                if prune.is_pruned(module): # Check if module was actually pruned
                    prune.remove(module, name)
            print("LLM pruning applied and made permanent.")
        else:
            print("No Linear modules found in LLM to prune.")

    if prune_vision_amount > 0 and hasattr(model, 'vision_tower'):
        print(f"Applying L1 unstructured pruning to Vision Encoder with amount: {prune_vision_amount}")
        parameters_to_prune_vision = []
        # Vision tower might be a list of modules or a single module
        vision_modules_to_check = model.vision_tower
        if not isinstance(vision_modules_to_check, torch.nn.ModuleList) and not isinstance(vision_modules_to_check, torch.nn.Sequential):
             vision_modules_to_check = [model.vision_tower]

        for vision_component in vision_modules_to_check:
            for module_name, module in vision_component.named_modules():
                if isinstance(module, torch.nn.Linear):
                    parameters_to_prune_vision.append((module, 'weight'))
        
        if parameters_to_prune_vision:
            prune.global_unstructured(
                parameters_to_prune_vision,
                pruning_method=prune.L1Unstructured,
                amount=prune_vision_amount,
            )
            # Make pruning permanent
            for module, name in parameters_to_prune_vision:
                 if prune.is_pruned(module): # Check if module was actually pruned
                    prune.remove(module, name)
            print("Vision Encoder pruning applied and made permanent.")
        else:
            print("No Linear modules found in Vision Encoder to prune.")
    return model

@torch.no_grad() 
def main(args):

    # Load model and processor
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(args.model_path)
    
    # model = PeftModel.from_pretrained(model, '/data2/private-data/zhangn/pretrained/paligemma/FULL-2025-04-29_21-11/final_model')
    # model = PeftModel.from_pretrained(model, '/data2/private-data/zhangn/pretrained/paligemma/FULL-2025-03-15_21-49/final_model')
    # model = model.merge_and_unload() # If using PeftModel and want to prune the base model

    if args.prune_llm_amount > 0 or args.prune_vision_amount > 0:
        print("Applying pruning...")
        model = apply_pruning(model, args.prune_llm_amount, args.prune_vision_amount)
        print("Pruning finished.")
    
    model.to(args.device)
    model.eval() # Ensure model is in evaluation mode

    generation_config = GenerationConfig.from_pretrained(args.model_path)

    # prepare dataset
    collate_fn_builder = build_collate_fn(args.collate_fn)
    val_collate_fn = partial(collate_fn_builder, processor=processor, dtype=torch.float16 if args.device == 'cuda' else torch.float32) # Adjust dtype based on device
    dataset = load_from_disk(args.data)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size, # Use batch_size from args
        collate_fn=val_collate_fn,
        num_workers=0, # Consider increasing num_workers if data loading is a bottleneck
        shuffle=False,
    )

    def timed_infer(inputs_on_device, generation_config_obj, target_max_new_tokens):
        input_len = inputs_on_device["input_ids"].shape[-1]
        
        # Time prefill + 1 token generation
        # This is an approximation of prefill time, as it includes one decoding step.
        torch.cuda.synchronize() # Ensure previous CUDA operations are done
        start_time_prefill = time.perf_counter()
        # Generate only 1 new token to estimate prefill + first token time
        _ = model.generate(
            **inputs_on_device,
            max_new_tokens=1, 
            generation_config=generation_config_obj,
            pad_token_id=processor.tokenizer.pad_token_id, # Ensure pad_token_id is set
            eos_token_id=processor.tokenizer.eos_token_id   # Ensure eos_token_id is set
        )
        torch.cuda.synchronize() # Ensure generation is done
        end_time_prefill = time.perf_counter()
        prefill_and_first_token_time = end_time_prefill - start_time_prefill

        # Time full generation
        torch.cuda.synchronize()
        start_time_full = time.perf_counter()
        output = model.generate(
            **inputs_on_device,
            max_new_tokens=target_max_new_tokens,
            generation_config=generation_config_obj,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
        torch.cuda.synchronize()
        end_time_full = time.perf_counter()
        full_generation_time = end_time_full - start_time_full
        
        generated_output = output[:, input_len:]
        results = processor.batch_decode(generated_output, skip_special_tokens=True)
        return results, prefill_and_first_token_time, full_generation_time

    def flatten(x):
        # Handles both batch_size=1 and actual batch processing
        if isinstance(x, list) and len(x) == 1 and args.batch_size == 1:
            return x[0]
        return x # Return list if batch_size > 1 or if it's already a single item
    
    data_dict = []
    all_prefill_times = []
    all_full_generation_times = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            inputs, questions, ids = batch # Assuming collate_fn returns these three
            
            inputs_on_device = {k: v.to(args.device) for k, v in inputs.items()}
            
            results, prefill_time, full_time = timed_infer(inputs_on_device, generation_config, args.max_new_tokens)
            
            all_prefill_times.append(prefill_time)
            all_full_generation_times.append(full_time)

            # Process results for each item in the batch
            batch_ids = ids if isinstance(ids, list) else [ids] # Ensure ids is a list
            batch_questions = questions if isinstance(questions, list) else [questions]
            batch_results = results if isinstance(results, list) else [results]

            for i in range(len(batch_ids)):
                data_dict.append({
                    'id': flatten(batch_ids[i]), 
                    'question': flatten(batch_questions[i]), 
                    'answer': flatten(batch_results[i])
                })

            # Save incrementally or at the end
            if args.output: # Only write if output path is provided
                 with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(data_dict, f, indent=4, ensure_ascii=False)

    if args.output:
        print(f"Inference results saved to {args.output}")

    if all_prefill_times:
        avg_prefill_time = np.mean(all_prefill_times)
        avg_full_generation_time = np.mean(all_full_generation_times)
        print(f"Average Prefill + 1 Token Generation Time: {avg_prefill_time:.4f} seconds")
        print(f"Average Full Generation Time ({args.max_new_tokens} new tokens): {avg_full_generation_time:.4f} seconds")
        print(f"Processed {len(all_prefill_times) * args.batch_size} samples.")
    else:
        print("No samples processed or timing data collected.")

def parse_args():
    parser = argparse.ArgumentParser(description='DriveLM Inference with Pruning and Timing')
    parser.add_argument("--model_path", type=str, default="lykong/paligemma-finetuned", help="Path to the pretrained model or model identifier from Huggingface Hub")
    parser.add_argument("--data", type=str, default="data/DriveLM_nuScenes/split/val", help="Path to the dataset")
    parser.add_argument("--collate_fn", type=str, default="drivelm_nus_paligemma_collate_fn_val", help="Name of the collate function to build")
    parser.add_argument("--output", type=str, default="data/DriveLM_nuScenes/refs/infer_results_timed.json", help="Path to save the inference results")
    parser.add_argument("--device", default="cuda", help="Device to run inference (e.g., 'cuda', 'cpu')")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum new tokens to generate for the main task") # Reduced default for faster testing
    
    parser.add_argument("--prune_llm_amount", type=float, default=0.0, help="Amount of pruning for LLM (0.0 to 1.0). 0.0 means no pruning.")
    parser.add_argument("--prune_vision_amount", type=float, default=0.0, help="Amount of pruning for Vision Encoder (0.0 to 1.0). 0.0 means no pruning.")
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print("Starting inference with arguments:", args)
    main(args)