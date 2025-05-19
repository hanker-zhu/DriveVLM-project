#!/bin/bash

# Script for automated testing of inference.py with different pruning configurations

# --- Configuration ---
PYTHON_EXE="python" # Or "python3"
INFERENCE_SCRIPT="tools/inference-my.py"
MODEL_PATH="lykong/paligemma-finetuned"
# Assuming default --data and --collate_fn from inference-my.py are acceptable for this test
# If not, add them here:
# DATA_PATH="data/DriveLM_nuScenes/split/val"
# COLLATE_FN="drivelm_nus_paligemma_collate_fn_val"

DEVICE="cuda"
BATCH_SIZE=1
MAX_NEW_TOKENS=100
RESULTS_DIR="results/automated_tests" # Directory to store output files

# Create results directory if it doesn't exist
mkdir -p "${RESULTS_DIR}"

# --- Helper Function to Run Inference ---
run_inference() {
    local p_llm_amount="$1"
    local p_vision_amount="$2"
    local test_name="llm_prune_${p_llm_amount}_vision_prune_${p_vision_amount}"
    local output_file="${RESULTS_DIR}/inference_${test_name}.json"
    local log_file="${RESULTS_DIR}/log_${test_name}.txt"

    echo "----------------------------------------------------------------------"
    echo "Running Test: ${test_name}"
    echo "LLM Pruning: ${p_llm_amount}, Vision Pruning: ${p_vision_amount}"
    echo "Output File: ${output_file}"
    echo "Log File: ${log_file}"
    echo "----------------------------------------------------------------------"

    command_to_run="${PYTHON_EXE} ${INFERENCE_SCRIPT} \
        --model_path ${MODEL_PATH} \
        --output ${output_file} \
        --device ${DEVICE} \
        --batch_size ${BATCH_SIZE} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --prune_llm_amount ${p_llm_amount} \
        --prune_vision_amount ${p_vision_amount}"
        # Add these if you defined DATA_PATH and COLLATE_FN above
        # --data ${DATA_PATH} \
        # --collate_fn ${COLLATE_FN} \

    echo "Executing: ${command_to_run}"

    # Execute and redirect stdout/stderr to a log file
    ${command_to_run} > "${log_file}" 2>&1

    if [ $? -eq 0 ]; then
        echo "Test ${test_name} COMPLETED successfully."
        echo "Results saved to ${output_file}"
        echo "Log saved to ${log_file}"
    else
        echo "Test ${test_name} FAILED. Check log: ${log_file}"
    fi
    echo "----------------------------------------------------------------------"
    echo ""
}

# --- Test Cases ---

# Test 1: LLM pruning 40%, no vision pruning
run_inference 0.4 0.0

# Test 2: LLM pruning 20%, no vision pruning (as in the original command)
run_inference 0.2 0.0

# Test 3: No pruning (baseline)
run_inference 0.0 0.0

# Test 4: Both LLM and Vision pruning 20%
run_inference 0.2 0.2

# Test 5: Vision pruning 20%, no LLM pruning
run_inference 0.0 0.2


# Add more test cases as needed:
# run_inference <llm_prune_value> <vision_prune_value>

echo "All automated tests finished."
echo "Check the ${RESULTS_DIR} directory for outputs and logs."
