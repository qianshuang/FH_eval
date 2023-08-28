# -*- coding: utf-8 -*-

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("ClueAI/ChatYuan-large-v2")
model = T5ForConditionalGeneration.from_pretrained("ClueAI/ChatYuan-large-v2").half()
device = torch.device('cuda')
model.to(device)

text = "你是谁？"
encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=1024, return_tensors="pt").to(device)
out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=1024, do_sample=True, top_p=1, temperature=0, no_repeat_ngram_size=12)
out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
print(out_text)
