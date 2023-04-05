import json
import math
from ipdb import set_trace
import jsonlines
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Union, Optional
from tqdm import tqdm

import torch
import random
import torch.distributed
import torch.nn.functional as F
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import SentencePieceBPETokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from sentencepiece import SentencePieceProcessor
import os


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        if self.eos_id in t:
            t = t[: t.index(self.eos_id) + 1]
        return self.sp_model.decode(t)


class HFLikeTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

        # assign attributes from real tokenizer to masked one
        self.pad_id = self.tokenizer.pad_id
        self.eos_id = self.tokenizer.eos_id
        self.bos_id = self.tokenizer.bos_id

        # mask attribute to be similar to hugging face
        self.eos_token_id = self.tokenizer.eos_id
        self.pad_token_id = self.tokenizer.pad_id

        # to match hugging face attribute
        self.pad_token_id = self.pad_id

    def create_sequence_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = torch.where(
            tokens == self.tokenizer.pad_id,
            torch.zeros_like(tokens),
            torch.ones_like(tokens),
        )
        mask = torch.where(tokens == self.tokenizer.bos_id, torch.zeros_like(tokens), mask)
        mask = torch.where(tokens == self.tokenizer.eos_id, torch.zeros_like(tokens), mask)
        return mask

    def __call__(self, texts: Union[List[str], str], *args, **kwargs):
        if isinstance(texts, str):
            text = self.tokenizer.encode(texts, bos=True, eos=True)
            tokens = torch.tensor(text).long()
            mask = torch.ones_like(tokens)
        else:
            texts = [self.tokenizer.encode(text, bos=True, eos=True) for text in texts]
            max_len = max(len(text) for text in texts)
            tokens = torch.full((len(texts), max_len), self.tokenizer.pad_id).long()
            for i, text in enumerate(texts):
                tokens[i, -len(text) :] = torch.tensor(text).long()  # noqa E203

            # TODO: decide how eos and bos should be handled - i need to mask
            # them? or not?
            mask = self.create_sequence_mask(tokens)
            for i in range(tokens.shape[0]):
                current_tokens = tokens[i, mask[i] == 1]
                tokens[i, -len(current_tokens) - 1 : -1] = current_tokens  # noqa E203
            mask = self.create_sequence_mask(tokens)

            # convert `pad_id` from -1 to 0, otherwise embedding will cause out
            # of bounds.
            tokens = torch.where(
                tokens == self.tokenizer.pad_id,
                torch.zeros_like(tokens),
                tokens,
            )
        output = {
            "input_ids": tokens,
            "attention_mask": mask,
        }
        return output

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)


class GSM8KDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        assert split in ("train", "test")
        data_path = os.path.join(data_dir, f"{split}.jsonl")
        self.questions, self.answers = self.load_data(data_path)

    def load_data(self, data_path):
        questions = []
        answers = []
        with open(data_path, "r") as f:
            for line in f:
                item = json.loads(line)
                questions.append(item["question"])
                answers.append(item["answer"])
        return questions, answers

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]
        answer = self.answers[index]
        return question, answer

    def get_data_batches(self, batch_size, shuffle=True):
        def collate_fn(batch):
            result = {"Question": [], "Answer": [], "Question_Answer": []}
            for b in batch:
                result["Question"].append("[Question]: {}\n[Answer]: ".format(b[0]))
                result["Answer"].append(b[1])
                result["Question_Answer"].append("[Question]: {}\n[Answer]: {}".format(b[0], b[1]))
            return result

        # Move the generator to the "cuda" device
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, generator=generator)


def main():
    tokenizer_path = "/remote-home/share/llama/tokenizer.model"
    tokenizer = Tokenizer(model_path=tokenizer_path)
    tokenizer = HFLikeTokenizer(tokenizer)

    train_dataset = GSM8KDataset("grade-school-math-master/grade_school_math/data", split="train")
    test_dataset = GSM8KDataset("grade-school-math-master/grade_school_math/data", split="test")

    train_dataloader = train_dataset.get_data_batches(batch_size=2, shuffle=True)
    test_dataloader = test_dataset.get_data_batches(batch_size=4, shuffle=True)

    for batch in train_dataloader:
        print(batch)
        input_ids = tokenizer(batch["Question_Answer"], return_tensors="pt", padding=True, truncation=True)
        set_trace()


if __name__ == "__main__":
    main()
