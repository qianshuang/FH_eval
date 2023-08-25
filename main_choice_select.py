# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score
from prompt_helper import *
from openai_helper import *

labels = []
predict_labels = []

file_path = "data/c3_public/d-dev.json"
with open(file_path, 'r') as file:
    json_object = json.load(file)
    print(len(json_object))

    for conv in json_object:
        conv_list = conv[0]
        for qa in conv[1]:
            question = qa["question"]
            answer = qa["answer"]
            choice_list = qa["choice"]

            labels.append(answer)
            prompt = gen_choice_select_prompt("\n".join(conv_list), question, choice_list)
            print(prompt[1]["content"])

            chat_res = get_chatgpt_res(prompt)
            chat_res = parse_json(chat_res)
            print(chat_res)
            print("----------------")

            predict_labels.append(chat_res["result"])

print("eval score: {}".format(accuracy_score(labels, predict_labels)))
