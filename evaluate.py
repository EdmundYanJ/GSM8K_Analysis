import re
import json
from ipdb import set_trace


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as fin:
        return [json.loads(_) for _ in fin.readlines()]


def load_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    data_list = []
    for line in lines:
        answer_index = line.find(", 'answer': ")
        answer_length = len(", 'answer': ")
        output_index = line.find(", 'output': ")
        output_length = len(", 'output': ")

        question = line[14 : answer_index - 1]
        answer = line[answer_index + answer_length + 1 : output_index - 1]
        output = line[output_index + output_length + 1 : -3]
        # if question == "" or answer == "" or output == "":
        #     set_trace()
        data_list.append({"question": question, "answer": answer, "output": output})

    return data_list


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


if __name__ == "__main__":
    data_list = load_txt("log/test_epoch_2_log_1.txt")
    label_list = load_json("grade-school-math-master/grade_school_math/data/test.jsonl")
    length = len(label_list)
    acc = 0
    for i in range(length):
        acc += evaluate_GSM8K(data_list[i]["output"], label_list[i]["answer"])
    print(acc / length)
