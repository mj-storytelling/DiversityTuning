import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

import evaluate
from evaluate import logging

def transform_scores(scores, min_score=None, max_score=None, compression_factor=100):
    """
    Transform scores to [-1, 1] range with configurable compression.
    
    Parameters:
    scores (array-like): Input scores
    min_score (float): Minimum score in the dataset (if None, will use min of scores)
    max_score (float): Maximum score in the dataset (if None, will use max of scores)
    method (str): Transformation method ('log', 'double_log', 'power')
    compression_factor (float): Controls compression strength (higher = more compression)
        - For 'log': number of times to apply log transformation
        - For 'power': exponent to use (should be between 0 and 1)
    
    Returns:
    array-like: Transformed scores in [-1, 1] range
    """
    scores = np.array(scores)
    
    # Use provided min/max or compute from data
    min_score = min_score if min_score is not None else scores.min()
    max_score = max_score if max_score is not None else scores.max()
    
    # Shift scores to be positive
    shifted_scores = scores - min_score + 1  # Add 1 to avoid log(0)

    min_trans = min_score
    max_trans = max_score
    
    # Apply log transformation multiple times based on compression_factor
    transformed = shifted_scores.copy()
    for _ in range(int(compression_factor)):
        transformed = np.log(transformed + 1)  # Add 1 to avoid log(0)
        min_trans = np.log(min_trans + 1)
        max_trans = np.log(max_trans + 1)
    
    normalized = 2 * (transformed - min_trans) / (max_trans - min_trans) - 1
    
    return normalized

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=False):
        self.patience = patience  # How many epochs to wait before stopping
        self.min_delta = min_delta  # Minimum change to qualify as an improvement
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model_indicator):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model_indicator
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping - best is model", self.best_model)
        else:
            self.best_loss = val_loss
            self.best_model = model_indicator
            self.counter = 0

class FlattenedDataset(Dataset):
    """
    Custom Dataset class to handle data and label pairs
    """
    def __init__(self, data, labels):
      self.data = data
            
      if not isinstance(labels, torch.Tensor):
          self.labels = torch.FloatTensor(np.array(labels)).to(torch.bfloat16)
      else:
          self.labels = labels
      # check if there is no NaN or infinitity in the labels
      if torch.isnan(self.labels).any() or torch.isinf(self.labels).any():
        raise ValueError("Labels should not contain NaN or infinity")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def save_to_disk(self, path):
      with open(path, 'wb') as f:
        save = {
          "data": self.data,
          "labels": self.labels.to(torch.float32).numpy(),
        }
        pickle.dump(save, f)
    
    @classmethod
    def load_from_disk(cls, path):
      with open(path, 'rb') as f:
        save = pickle.load(f)
        return cls(save["data"], save["labels"])


_CITATION = """\

"""

_DESCRIPTION = """
Perplexity (PPL) is one of the most common metrics for evaluating language models.
It is defined as the exponentiated average negative log-likelihood of a sequence, calculated with exponent base `e`.

For more information, see https://huggingface.co/docs/transformers/perplexity
"""

_KWARGS_DESCRIPTION = """
Args:
    model_id (str): model used for calculating Perplexity
            NOTE: Perplexity can only be calculated for causal language models.
                    This includes models such as gpt2, causal variations of bert,
                    causal versions of t5, and more (the full list can be found
                    in the AutoModelForCausalLM documentation here:
                    https://huggingface.co/docs/transformers/master/en/model_doc/auto#transformers.AutoModelForCausalLM )

    predictions (list of str): input text, each separate text snippet
        is one list entry.
    batch_size (int): the batch size to run texts through the model. Defaults to 16.
    add_start_token (bool): whether to add the start token to the texts,
        so the perplexity can include the probability of the first word. Defaults to True.
    device (str): device to run on, defaults to 'cuda' when available
Returns:
    perplexity: dictionary containing the perplexity scores for the texts
        in the input list, as well as the mean perplexity. If one of the input texts is
        longer than the max input length of the model, then it is truncated to the
        max length for the perplexity computation.
Examples:
    Example 1:
        >>> perplexity = evaluate.load("perplexity", module_type="metric")
        >>> input_texts = ["lorem ipsum", "Happy Birthday!", "Bienvenue"]
        >>> results = perplexity.compute(model_id='gpt2',
        ...                              add_start_token=False,
        ...                              predictions=input_texts) # doctest:+ELLIPSIS
        >>> print(list(results.keys()))
        ['perplexities', 'mean_perplexity']
        >>> print(round(results["mean_perplexity"], 0))
        647.0
        >>> print(round(results["perplexities"][0], 0))
        32.0

    Example 2:
        >>> from datasets import load_dataset
        >>> perplexity = evaluate.load("perplexity", module_type="metric")
        >>> input_texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"][:10] # doctest: +SKIP
        >>> input_texts = [s for s in input_texts if s!='']
        >>> results = perplexity.compute(model_id='gpt2',
        ...                              predictions=input_texts)
        >>> print(list(results.keys()))
        ['perplexities', 'mean_perplexity']
        >>> print(round(results["mean_perplexity"], 2)) # doctest: +SKIP
        576.76
        >>> print(round(results["perplexities"][0], 2)) # doctest: +SKIP
        889.28
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Perplexity(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation= _CITATION,
            # inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                }
            ),
            reference_urls=["https://huggingface.co/docs/transformers/perplexity"],
        )

    def _compute(
        self, predictions, model_id=None, modelNtokenizer=None, batch_size: int = 16, add_start_token: bool = True, device=None, max_length=None
    ):

        if device is not None:
            if "cuda" not in device:
                assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
            if device == "gpu":
                device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_id:
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model = model.to(device)

            tokenizer = AutoTokenizer.from_pretrained(model_id)
        elif modelNtokenizer:
            model = modelNtokenizer['model']
            tokenizer = modelNtokenizer['tokenizer']
        else:
            raise ValueError("Either model_id or modelNtokenizer should be provided.")

        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


import random
from typing import List, Tuple

def sample_pairs_with_gap(numbers: List[int], num_pairs: int = -1, min_gap: int = 1) -> List[Tuple[int, int]]:
    """
    Sample pairs of different integers from a list, attempting to maintain a minimum gap between paired numbers.
    
    Args:
        numbers: List of integers to sample from
        num_pairs: Number of pairs to sample
        min_gap: Minimum desired gap between numbers in a pair (if possible)
    
    Returns:
        List of tuples containing the sampled pairs
    """
    if len(numbers) < 2:
        return [], []
    
    # Make a copy and sort the numbers
    number_with_index = [(num, idx) for idx, num in enumerate(numbers)]
    sorted_nums = sorted(number_with_index, key=lambda x: x[0])

    available_nums = [numset for numset in sorted_nums]
    pairs = []
    
    while len(available_nums) >= 2:
        if num_pairs > 0 and len(pairs) >= num_pairs :
            break
        # Try to find pairs with minimum gap first
        valid_pairs = []
        for nidx1 in range(len(available_nums)):
            for nidx2 in range(nidx1 + 1, len(available_nums)):
                num1set = available_nums[nidx1]
                num2set = available_nums[nidx2]
                if num1set[0] != num2set[0] and abs(num1set[0] - num2set[0]) >= min_gap:
                    valid_pairs.append((num1set, num2set))
        
        # If no pairs with minimum gap are available, fall back to any valid pair
        if not valid_pairs:
            valid_pairs = [(available_nums[nidx1], available_nums[nidx2]) 
                          for nidx1 in range(len(available_nums))
                          for nidx2 in range(nidx1 + 1, len(available_nums))
                          if available_nums[nidx1][0] != available_nums[nidx2][0]]
        
        if not valid_pairs:
            break
            
        # Randomly select a pair
        chosen_pair = random.choice(valid_pairs)
        pairs.append(chosen_pair)

        removed_idxs = [chosen_pair[0][1], chosen_pair[1][1]]

        # Remove used numbers
        available_nums = [numset for numset in available_nums if numset[1] not in removed_idxs]
    
    return pairs, available_nums


from peft import PeftModel, PeftConfig, LoraConfig
import torch

def copy_peft_adapter(loaded_peft_model, new_adapter_name):
    """
    Copy an existing PEFT adapter to create a new adapter with a different name.
    
    Args:
        base_model: The original base model
        loaded_peft_model: The PEFT model with the adapter you want to copy
        new_adapter_name: Name for the new adapter copy
    
    Returns:
        PeftModel with both the original and copied adapter
    """
    # Get the original adapter name
    original_adapter_name = loaded_peft_model.active_adapter
    
    # Get the configuration of the original adapter
    original_config = loaded_peft_model.peft_config[original_adapter_name]
    
    # # Create a new config for the copied adapter
    # new_config = PeftConfig(
    #     peft_type=original_config.peft_type,
    #     task_type=original_config.task_type,
    #     inference_mode=original_config.inference_mode,
    #     r=original_config.r,
    #     lora_alpha=original_config.lora_alpha,
    #     lora_dropout=original_config.lora_dropout,
    #     bias=original_config.bias,
    #     target_modules=original_config.target_modules
    # )
    new_config = LoraConfig(
        r = original_config.r,
        lora_alpha = original_config.lora_alpha,
        bias = original_config.bias,
        task_type = original_config.task_type,
    )
    
    # Add the new adapter to the model
    loaded_peft_model.add_adapter(new_adapter_name, new_config)
    
    # Copy the weights from the original adapter to the new one
    with torch.no_grad():
        for name, param in loaded_peft_model.state_dict().items():
            if original_adapter_name in name:
                new_name = name.replace(original_adapter_name, new_adapter_name)
                loaded_peft_model.state_dict()[new_name].copy_(param)
    
    return loaded_peft_model