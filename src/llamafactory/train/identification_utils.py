from transformers import AutoModelForCausalLM, AutoTokenizer
from captum.attr import LayerIntegratedGradients
from tqdm import tqdm

def identify_important_layers(model, data, tokenizer, device):
    """
    Identify and rank the model layers by their importance for answering questions.

    Args:
        model: The language model.
        data: A list of tuples (question, correct_answer). Extra elements in each tuple are ignored.
        tokenizer: The tokenizer to use for encoding inputs.
        device: The device on which to run the computations.

    Returns:
        A list of layer indices sorted in descending order of average importance.
    """
    model.eval()

    # --- Optionally try to set static graph ---
    if hasattr(model, "_set_static_graph"):
        try:
            model._set_static_graph()
        except Exception as e:
            logger.warning(f"Unable to set static graph: {e}")

    # Dictionary to accumulate importance scores per layer index
    layer_scores = {}

    # Loop over all data entries to accumulate scores
    for entry in data:
        question, correct_str = entry[:2]
        encoded_input = tokenizer(question, return_tensors="pt")
        input_ids = encoded_input["input_ids"].to(device)

        correct_answer_ids = tokenizer(correct_str, add_special_tokens=False)["input_ids"]
        if len(correct_answer_ids) != 1:
            raise ValueError(
                f"Answer '{correct_str}' tokenizes into multiple tokens {correct_answer_ids}. "
                "For a multi-token answer, you need a different approach or to sum across those tokens."
            )
        correct_token_id = correct_answer_ids[0]

        # Determine layers from the model
        if hasattr(model, "layers"):
            layers = model.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        else:
            raise ValueError("Model does not have layers attribute.")

        # Define forward function capturing the current input and token id.
        def forward_func(input_ids_):
            outputs = model(input_ids=input_ids_)
            last_token_logits = outputs.logits[:, -1, :]
            return last_token_logits[:, correct_token_id]

        # Compute importance for each layer on the current input
        for i, layer in enumerate(layers):
            torch.cuda.empty_cache()  # Clear cache before computation
            with torch.cuda.amp.autocast():
                lig = LayerIntegratedGradients(forward_func, layer)
                baselines = torch.full_like(input_ids, tokenizer.pad_token_id, dtype=torch.long)
                # Remove convergence delta to reduce backward calls.
                attributions = lig.attribute(
                    inputs=input_ids,
                    baselines=baselines,
                    n_steps=10,
                    return_convergence_delta=False
                )
            importance_score = attributions.abs().sum().item()
            layer_scores.setdefault(i, []).append(importance_score)
            # Clear gradients to prevent reusing the same graph in successive backward calls.
            model.zero_grad()
            del lig, baselines, attributions
            torch.cuda.empty_cache()

    # Average the importance scores over all samples for each layer
    avg_importance = {i: sum(scores) / len(scores) for i, scores in layer_scores.items()}
    # Sort layers by descending average importance
    sorted_layers = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    return [i for i, score in sorted_layers]