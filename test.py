# -*- coding: utf-8 -*-

from transformers import AutoModelForSequenceClassification

model_name = "xlm-roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
