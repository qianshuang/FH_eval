# -*- coding: utf-8 -*-

from prompt_helper import *
from openai_helper import *

predict_labels = []

file_path = "data/cmrc2018_public/dev.json"
with open(file_path, 'r') as file:
    json_object = json.load(file)
    print(len(json_object["data"]))

    for p in json_object["data"]:
        for pa in p["paragraphs"]:
            context = pa["context"]
            for qa in pa["qas"]:
                question = qa["question"]
                answers = [an["text"] for an in qa["answers"]]

                prompt = gen_answer_extract_prompt(context, question)
                print(prompt[1]["content"])

                chat_res = get_chatgpt_res(prompt)
                chat_res = parse_json(chat_res)
                print(chat_res)

                # TODO 这里使用语义相似度更加合适
                if "result" in chat_res and str(chat_res["result"]) in answers:
                    predict_labels.append(1)
                else:
                    predict_labels.append(0)
                print("current eval score: {}".format(predict_labels.count(1) / len(predict_labels)))

print("final eval score: {}".format(predict_labels.count(1) / len(predict_labels)))  # 0.2480916030534351
