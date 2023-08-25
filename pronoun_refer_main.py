# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score
from prompt_helper import *
from openai_helper import *

labels = []
predict_labels = []

# 1. 数据加载
jsonl_file_path = "data/cluewsc2020_public/dev.json"
with open(jsonl_file_path, 'r') as jsonl_file:
    for line in jsonl_file:
        print(line)
        json_object = json.loads(line)

        text = json_object["text"]
        label = json_object["label"]
        pronoun = json_object["target"]["span2_text"]
        refer = json_object["target"]["span1_text"]

        labels.append(label)
        prompt = gen_prompt(text, pronoun, refer)
        print(prompt[1]["content"])

        chat_res = get_chatgpt_res(prompt)
        chat_res = parse_json(chat_res)
        print(chat_res)

        predict_labels.append(str(chat_res["result"]).lower())

print("eval score: {}".format(accuracy_score(labels, predict_labels)))
