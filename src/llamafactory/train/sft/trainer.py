import json
import os
from types import MethodType
from typing import Any, Optional

import torch
import numpy as np  # <- Make sure you have this if you're decoding predictions below
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.

    This custom trainer is also able to freeze specific neurons during training.
    You can specify a freeze configuration in two ways:
      1. Directly via a list of pairs in `finetuning_args.freeze_neurons_config`
         (e.g. [[layer_idx, neuron_idx], ...]).
      2. Via a JSON file provided in `finetuning_args.freeze_neurons_config_file`
         whose content contains one or more JSON objects with a `"neurons"` field.
         For example, a line in the file might look like:
         {"uuid": "0a6d35b2-1066-4af7-9a82-...", "neurons": [[0, 7345], [10, 6464], [13, 4681]], ...}
    """

    def __init__(
        self,
        finetuning_args: Any,
        processor: Optional[Any],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Handle tokenizer for different transformers versions
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        # Optionally split the training dataset if split_idx is provided in finetuning_args.
        # The dataset is split into three (roughly equal) parts and the split specified by split_idx is used.
        if hasattr(self.finetuning_args, "split_idx"):
            split_idx = self.finetuning_args.split_idx
            total = len(self.train_dataset)
            # Compute the sizes for each split (handling any remainder)
            split_sizes = [total // 3 + (1 if i < total % 3 else 0) for i in range(3)]
            splits = []
            start = 0
            for size in split_sizes:
                splits.append(self.train_dataset[start : start + size])
                start += size
            if not (0 <= split_idx < 3):
                raise ValueError("split_idx must be 0, 1, or 2.")
            self.train_dataset = splits[split_idx]
            logger.info(f"Using split {split_idx} of training dataset (length: {len(self.train_dataset)}).")

        # Process freeze-neuron configuration either from a direct list or from a JSON file.
        neuron_pairs = []

        # Option 1: Direct configuration
        if hasattr(finetuning_args, "freeze_neurons_config"):
            neuron_pairs.extend(finetuning_args.freeze_neurons_config)
            logger.info(
                f"Loaded direct neuron freeze configuration: {finetuning_args.freeze_neurons_config}"
            )

        # Option 2: Load configuration from a JSON file
        if hasattr(finetuning_args, "freeze_neurons_config_file"):
            file_path = finetuning_args.freeze_neurons_config_file
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                                if "neurons" in obj:
                                    neuron_pairs.extend(obj["neurons"])
                                    logger.info(
                                        f"Loaded neuron freeze configuration "
                                        f"from JSON object with uuid {obj.get('uuid', 'unknown')}: {obj['neurons']}"
                                    )
                                else:
                                    logger.warning(
                                        "JSON object in freeze_neurons_config_file "
                                        "lacks a 'neurons' field."
                                    )
                            except Exception as e:
                                logger.error(f"Error parsing line in freeze_neurons_config_file: {e}")
                except Exception as e:
                    logger.error(f"Error reading freeze_neurons_config_file: {e}")
            else:
                logger.error(f"Freeze neurons config file {file_path} does not exist.")

        # Freeze the specified neurons
        frozen_neurons_count = 0  # Track how many we actually manage to freeze
        if neuron_pairs:
            if hasattr(self.model, "layers"):
                for pair in neuron_pairs:
                    try:
                        layer_idx, neuron_idx = pair[0], pair[1]
                    except (IndexError, TypeError):
                        logger.warning(
                            f"Invalid neuron pair: {pair}. "
                            f"Expected format [layer_idx, neuron_idx]."
                        )
                        continue

                    if layer_idx < len(self.model.layers):
                        layer = self.model.layers[layer_idx]
                        self.freeze_neuron(layer, neuron_idx)
                        frozen_neurons_count += 1
                        logger.info(f"Freezing neuron {neuron_idx} in layer {layer_idx}.")
                    else:
                        logger.warning(
                            f"Layer index {layer_idx} is out of range "
                            f"(model has {len(self.model.layers)} layers)."
                        )
            else:
                logger.warning(
                    "Model does not have a 'layers' attribute. Neuron freezing skipped."
                )

            # Finally, log the total number of neurons that we attempted to freeze
            logger.info(f"Total neurons frozen: {frozen_neurons_count}")

        # (Optional) Other callbacks or modifications can be added below.
        if getattr(finetuning_args, "use_badam", False):
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            # Patch the accelerator’s clipping method
            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def freeze_neuron(self, module: torch.nn.Module, neuron_idx: int) -> None:
        """
        Attaches a backward hook to the given module's weight (and bias if present)
        so that the gradient for the specified neuron (i.e., a row in the weight) is zeroed out.
        """

        def weight_hook(grad: torch.Tensor) -> torch.Tensor:
            if neuron_idx < grad.size(0):
                grad[neuron_idx, :] = 0
            return grad

        module.weight.register_hook(weight_hook)

        if hasattr(module, "bias") and module.bias is not None:
            def bias_hook(grad: torch.Tensor) -> torch.Tensor:
                if neuron_idx < grad.size(0):
                    grad[neuron_idx] = 0
                return grad

            module.bias.register_hook(bias_hook)

    @override
    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional[torch.optim.Optimizer] = None
    ) -> torch.optim.lr_scheduler.LRScheduler:
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if getattr(self.finetuning_args, "disable_shuffling", False):
            return torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler()

    @override
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.args.predict_with_generate:
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            # Mask out the input_ids part of the generation to keep only the newly generated tokens
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: Any, predict_results: Any, skip_special_tokens: bool = True
    ) -> None:
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = predict_results.label_ids
        preds = predict_results.predictions

        # Replace IGNORE_INDEX with pad token id
        labels = np.where(labels != IGNORE_INDEX, labels, self.processing_class.pad_token_id)
        preds = np.where(preds != IGNORE_INDEX, preds, self.processing_class.pad_token_id)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(
                    json.dumps(
                        {"prompt": text, "predict": pred, "label": label},
                        ensure_ascii=False
                    ) + "\n"
                )
