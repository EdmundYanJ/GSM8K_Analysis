import time
import random
import transformers
import torch
import tqdm
import torch.cuda
import os
import json
import re
import numpy as np
from torch.utils.data import Dataset
import jsonlines
from ipdb import set_trace
from typing import Tuple, List, Union, Optional
from tqdm import tqdm

torch.manual_seed(42)
random.seed(42)
os.environ["OPT_st_layer"] = "0"
os.environ["OPT_ed_layer"] = "999"
# model_name_or_path = "facebook/opt-iml-max-30b"
# model_name_or_path = "facebook/opt-66b"


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

    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx]

    def process_data(self, batch_size=4, shuffle=False):
        if shuffle:
            shuffled_indices = np.random.permutation(len(self.questions))
        else:
            shuffled_indices = range(len(self.questions))

        questions = [self.questions[i] for i in shuffled_indices]
        answers = [self.answers[i] for i in shuffled_indices]
        prompts = ["[Question]: {}\n[Answer]: ".format(self.questions[i]) for i in shuffled_indices]
        labels = ["[Question]: {}\n[Answer]: {}".format(self.questions[i], self.answers[i]) for i in shuffled_indices]
        batches = []

        for i in range(0, len(self.questions), batch_size):
            batch_questions = questions[i : i + batch_size]
            batch_answers = answers[i : i + batch_size]
            batch_prompt = prompts[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]
            batches.append((batch_questions, batch_answers, batch_prompt, batch_labels))
        batches.append((questions[i:], answers[i:], prompts[i:], labels[i:]))
        return batches


def evaluate_GSM8K(y_pred, y_label):
    pattern = r"\[Answer\]:\s*(.*?)\n(\[Question\]:|$)"
    match = re.search(pattern, y_pred)
    if match:
        y_pred = match.group(1)
    pred = y_pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]
    if pred == []:
        return 0
    pred = pred[-1]
    label = y_label.split("#### ")[-1]
    if pred == label:
        return 1
    else:
        return 0


def train():
    model_name_or_path = "facebook/opt-1.3b"
    version = "v1"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, cache_dir="/remote-home/share/llms", local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", cache_dir="/remote-home/share/llms", load_in_8bit=False, torch_dtype=torch.bfloat16, local_files_only=True)

    train_dataset = GSM8KDataset("grade-school-math-master/grade_school_math/data", split="train")
    test_dataset = GSM8KDataset("grade-school-math-master/grade_school_math/data", split="test")

    train_length = len(train_dataset)
    test_length = len(test_dataset)

    train_dataloader = train_dataset.process_data(batch_size=8, shuffle=True)
    test_dataloader = test_dataset.process_data(batch_size=8, shuffle=True)
    model.train()
    lr = 1e-4
    os.makedirs("opt_log/{}/{}".format(version, model_name_or_path), exist_ok=True)

    # An approximation of in-place grad update
    def func(x):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    # p.data -= (lr*p.grad.data+1e-5*p.data)
                    p.data -= lr * p.grad.data
                    p.grad = None
        return x

    for n, p in model.named_parameters():
        if p.requires_grad:
            p.register_hook(func)

    model.train()
    epoch = 0
    with torch.no_grad():
        acc = 0
        num = 0
        res = {"question": [], "answer": [], "prediction": [], "prompt": [], "label": []}
        with tqdm(test_dataloader, desc="Testing", leave=False) as tq:
            for batch in tq:
                questions, answers, prompts, labels = batch
                test_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                # input_ids attention_mask max_new_tokens temperature top_p no_repeat_ngram_size
                y_pred = model.generate(input_ids=test_prompts["input_ids"], attention_mask=test_prompts["attention_mask"], max_new_tokens=128, temperature=0.1, top_p=0.5, no_repeat_ngram_size=5)
                # print(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024)
                length = len(answers)
                for i in range(length):
                    pred = tokenizer.decode(y_pred[i][torch.sum(test_prompts["attention_mask"][i] == 0) - 1 :].tolist())
                    acc += evaluate_GSM8K(pred, answers[i])
                    num += 1
                    with jsonlines.open(os.path.join("opt_log/{}/{}".format(version, model_name_or_path), "epoch_{}_log.jsonl".format(epoch)), "a") as writer:
                        writer.write({"question": questions[i], "answer": answers[i], "prediction": pred, "prompt": prompts[i], "label": labels[i]})
                    res["question"].append(questions[i])
                    res["answer"].append(answers[i])
                    res["prediction"].append(pred)
                    res["prompt"].append(prompts[i])
                    res["label"].append(labels[i])
                tq.set_postfix({"acc": acc / num})
        with open(os.path.join("opt_log/{}/{}".format(version, model_name_or_path), "epoch_{}_res.json".format(epoch)), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
        print("\n**************Epoch:{} Acc: {}**************".format(epoch, acc / num))

    model.train()
    for epoch in range(1, 17):
        with tqdm(train_dataloader, desc="Training", leave=False) as tq:
            for batch in tq:
                questions, answers, prompts, labels = batch
                train_labels = tokenizer(labels, return_tensors="pt", padding=True, truncation=True)
                model.zero_grad()
                res = model(train_labels["input_ids"].cuda(), labels=train_labels["input_ids"].cuda())
                loss = res[0]
                loss.backward()
                func(0)
                tq.set_postfix({"loss": loss.item()})

        print("\n**************Epoch:{}**************".format(epoch))
        # test the model
        with torch.no_grad():
            acc = 0
            num = 0
            res = {"question": [], "answer": [], "prediction": [], "prompt": [], "label": []}
            with tqdm(test_dataloader, desc="Testing", leave=False) as tq:
                for batch in tq:
                    questions, answers, prompts, labels = batch
                    test_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                    # input_ids attention_mask max_new_tokens temperature top_p no_repeat_ngram_size
                    model.zero_grad()
                    y_pred = model.generate(input_ids=test_prompts["input_ids"], attention_mask=test_prompts["attention_mask"], max_new_tokens=128, temperature=0.1, top_p=0.5, no_repeat_ngram_size=5)

                    # print(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024)
                    length = len(answers)
                    for i in range(length):
                        pred = tokenizer.decode(y_pred[i].tolist())
                        acc += evaluate_GSM8K(pred, answers[i])
                        num += 1
                        with jsonlines.open(os.path.join("opt_log/{}/{}".format(version, model_name_or_path), "epoch_{}_log.jsonl".format(epoch)), "a") as writer:
                            writer.write({"question": questions[i], "answer": answers[i], "prediction": pred, "prompt": prompts[i], "label": labels[i]})
                        res["question"].append(questions[i])
                        res["answer"].append(answers[i])
                        res["prediction"].append(pred)
                        res["prompt"].append(prompts[i])
                        res["label"].append(labels[i])
                    tq.set_postfix({"acc": acc / num})
            with open(os.path.join("opt_log/{}/{}".format(version, model_name_or_path), "epoch_{}_res.json".format(epoch)), "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=4)
            print("\n**************Epoch:{} Acc: {}**************".format(epoch, acc / num))


if __name__ == "__main__":
    train()
