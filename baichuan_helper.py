# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-13B-Chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-13B-Chat", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan-13B-Chat")


def get_chatyuan_res(messages):
    response = model.chat(tokenizer, messages)
    print(response)


messages = [{"role": "system", "content": "You are a trustworthy AI assistant."},
            {"role": "user", "content": "世界上第二高的山峰是哪座？"}]
get_chatyuan_res(messages)
