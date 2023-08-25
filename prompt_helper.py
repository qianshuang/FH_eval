# -*- coding: utf-8 -*-


def gen_pronoun_refer_prompt(text, pronoun, refer):
    prompt_str = """GOAL:
Read and comprehend the given TEXT, judge whether the content referred to by the pronoun in the TEXT is correct.
Let's work this out in a step by step way to be sure we have the right answer.

TEXT:
{}

PRONOUN:
{}

Referred Content:
{}

You should only respond in JSON format as described below:
{{
    "reasoning": "reasoning",
    "result": "True if the content referred to by the PRONOUN in the TEXT is the given Referred Content, False otherwise."
}}
Ensure the response can be parsed by Python json.loads""".format(text, pronoun, refer)

    return [{"role": "system", "content": "You are a trustworthy AI assistant."},
            {"role": "user", "content": prompt_str}]


def gen_choice_select_prompt(conversation, question, choice_list):
    prompt_str = """GOAL:
Given the DIALOGUE and QUESTION below, choose the most correct one from CHOICES as the answer to the QUESTION.
Let's work this out in a step by step way to be sure we have the right answer.

DIALOGUE:
{}

QUESTION:
{}

CHOICES:
{}

You should only respond in JSON format as described below:
{{
    "reasoning": "reasoning",
    "result": "the most correct answer selected from CHOICES"
}}
Ensure the response can be parsed by Python json.loads""".format(conversation, question, choice_list)

    return [{"role": "system", "content": "You are a trustworthy AI assistant."},
            {"role": "user", "content": prompt_str}]
