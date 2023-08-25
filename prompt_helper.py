# -*- coding: utf-8 -*-


def gen_prompt(text, pronoun, refer):
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
