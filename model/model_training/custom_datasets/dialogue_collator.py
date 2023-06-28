import random
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from model_training.custom_datasets.formatting import (
    QA_SPECIAL_TOKENS,
    DatasetEntryLm,
    DatasetEntrySft,
    format_pairs,
    format_system_prefix,
)
from torch.nn import functional as F
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase, TruncationStrategy


@dataclass
class DialogueDataCollator:
    """
    Expects a list of texts corresponding to a sequence of [question, answer, question, answer, ...] pairs.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    mix_length_threshold: Optional[int] = 256
    mix_probability: Optional[float] = 1
    pad_to_multiple_of: Optional[int] = None
    samples_mixing: Optional[bool] = False
    random_offset_probability: Optional[float] = 0.5
    label_masking: bool = True
    use_system_prefix: bool = False
    system_prefix: str = None
    use_system_tag: bool = False
    system_property_dropout: float = 0.5
    system_add_length: bool = True

    def __post_init__(self):
        assert self.tokenizer.eos_token

        if self.use_system_prefix:
            assert self.system_prefix
            self.system_prefix = self.tokenizer.encode(
                format_system_prefix(self.system_prefix, self.tokenizer.eos_token),
                add_special_tokens=False,
                return_tensors="np",
            )[0]
            self.max_length = self.max_length - len(self.system_prefix)

    def process_one(self, messages, return_length=False):
        total_short_context_one = 0
        max_length = self.max_length

        pretrain_dataset = False
        if isinstance(messages, DatasetEntrySft):
            messages_formatted = messages.get_formatted(
                eos_token=self.tokenizer.eos_token,
                use_system_tag=self.use_system_tag,
                system_property_dropout=self.system_property_dropout,
                system_add_length=self.system_add_length,
            )
        else:
            raise ValueError("DatasetEntrySft expected")
        
        label_mask = []
        token_ids = [] 
        messages_formatted = [x + "\n" if i != len(messages_formatted) - 1 else x + "<|endoftext|>" for (i, x) in enumerate(messages_formatted)]
        token_ids_pre = self.tokenizer.batch_encode_plus(messages_formatted)["input_ids"]
        if self.label_masking:
            count = 0
            for (mess, tokens) in zip(messages.conversation, token_ids_pre):
                count += 1
                if mess.role == "assistant":
                    label_mask.extend([1] * len(tokens))
                    token_ids.extend(tokens)
                else:
                    label_mask.extend([0] * len(tokens))
                    token_ids.extend(tokens)
        attention_mask = [1] * len(token_ids)
        assert len(token_ids) == len(label_mask) == len(attention_mask)
        if len(token_ids) > max_length:
            token_ids = token_ids[-max_length:]
            label_mask = label_mask[-max_length:]
            attention_mask = attention_mask[-max_length:]
            
        return token_ids, attention_mask, label_mask 

    def __call__(self, features):
        flatten_messages = []
        label_masks = []
        attention_masks = []
        for messages in features:
            flatten_message, attention_mask, label_mask = self.process_one(messages)
            flatten_messages.append(flatten_message)
            label_masks.append(label_mask)
            attention_masks.append(attention_mask)
            
        batch = {"input_ids": flatten_messages, "attention_mask": attention_masks}
        batch = self.tokenizer.pad(
            batch,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["targets"] = torch.roll(batch.input_ids, -1, -1)
        dim = batch.input_ids.shape[-1]
        batch["label_masks"] = torch.stack(
            [F.pad(torch.tensor(x), (0, dim - len(x)), value=False) for x in label_masks]
        )
        return batch
