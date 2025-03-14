import json
import re
from datasets import load_dataset

def format_data(example):
    """
    Formats a single MathQA example into the LLaMA-Factory fine-tuning format.

    Args:
        example (dict): A single data sample from the dataset.

    Returns:
        dict: Processed data containing 'instruction', 'input', and 'output'.
    """
    problem = example["Problem"]
    choices = example["options"].strip()
    rationale = example["Rationale"]
    correct_answer = example["correct"].upper()

    # Format choices
    raw_choices = [c.strip() for c in choices.split(",")]
    formatted_choices = "\n".join(
        [re.sub(r"([a-eA-E])\)", r"(\1)", choice) for choice in raw_choices]
    )

    return {
        "instruction": (
            "Solve the following math problem. Show your reasoning step by step "
            "and output the answer choice at last. Format your answer choice exactly as follows:\n"
            " Answer: (X)  # where X is one of A, B, C, D, or E\n\n"
        ),
        "input": f"Question: {problem}\nOptions:\n{formatted_choices}",
        "output": f"{rationale} \n Answer: ({correct_answer})"
    }


def main():
    # Load MathQA dataset and extract only required fields
    dataset = load_dataset("allenai/math_qa", split="train")
    formatted_data = [format_data(example) for example in dataset]

    # Save as JSON
    output_file = "mathqa_finetune_formatted.json"
    with open(output_file, "w") as f:
        json.dump(formatted_data, f, indent=4)

    print(f"Dataset saved as '{output_file}'.")

if __name__ == "__main__":
    main()
