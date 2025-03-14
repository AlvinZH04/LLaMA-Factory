import json
import random
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams

# Define paths
CHECKPOINT_PATH = "/scratch/dkhasha1/bzhang90/LLaMA-Factory/saves/llama3-3b/freeze/sft_first20_formatted/checkpoint-1864"
OUTPUT_FILE = "mathqa_inference_results_vllm_sft_first20_formattuned.json"
NUM_SAMPLES = 100  # Number of test samples to evaluate
BATCH_SIZE = 50  # Number of queries to process in parallel
SEED = 42  # Set random seed for reproducibility

# Load MathQA test set
print("Loading dataset...")
test_dataset = load_dataset("allenai/math_qa", split="test")

# Set random seed for reproducibility
random.seed(SEED)

# Randomly select a subset
test_samples = random.sample(list(test_dataset), NUM_SAMPLES)

# Load vLLM model
print("Loading model with vLLM...")
llm = LLM(model=CHECKPOINT_PATH, tensor_parallel_size=4)  # Adjust based on GPU setup

# Sampling parameters (modify if needed)
sampling_params = SamplingParams(
    temperature=0.0,  
    #repetition_penalty = 2.0,
    max_tokens=1536
)

# Prepare inference and results
results = []

print("Running inference in batches...")

# Split into batches
batched_samples = [test_samples[i:i + BATCH_SIZE] for i in range(0, len(test_samples), BATCH_SIZE)]

for batch in tqdm(batched_samples, desc="Processing Batches", unit="batch"):
    prompts = []
    metadata = []  # To keep track of the corresponding questions

    # Create batch of prompts
    for sample in batch:
        problem = sample["Problem"]
        choices = sample["options"].strip()

        # Format choices
        raw_choices = [c.strip() for c in choices.split(",")]
        formatted_choices = "\n".join(raw_choices)

        # Construct the prompt
        prompt = (
            "Solve the following math problem. Show your reasoning step by step "
            "and output the answer choice at last. Format your answer choice exactly as follows:\n"
            "Answer: (X)  # where X is one of A, B, C, D, or E\n\n"
            f"Question: {problem}\n"
            f"Options:\n{formatted_choices}\n\n"
        )

        prompts.append(prompt)
        metadata.append({
            "question": problem,
            "choices": formatted_choices,
            "expected_answer": f"({sample['correct']})"
        })

    # Run batched inference using vLLM
    outputs = llm.generate(prompts, sampling_params)

    # Process results
    for meta, output in zip(metadata, outputs):
        model_response = output.outputs[0].text  # Extract generated text

        results.append({
            "question": meta["question"],
            "choices": meta["choices"],
            "expected_answer": meta["expected_answer"],
            "model_response": model_response
        })

# Save results to JSON
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=4)

print(f"Inference completed. Results saved to {OUTPUT_FILE}")
