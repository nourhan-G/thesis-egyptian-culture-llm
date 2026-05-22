import os
import re
import json
import csv
import pandas as pd
from datetime import datetime
from openai import OpenAI
from anthropic import Anthropic
from huggingface_hub import InferenceClient

OPENAI_API_KEY = "KEY"
client = OpenAI(api_key=OPENAI_API_KEY)


CLAUDE_API_KEY = "KEY"
claude_client = Anthropic(api_key=CLAUDE_API_KEY)

HF_TOKEN = "KEY"

hf_client = InferenceClient(provider="auto",api_key=HF_TOKEN)

def run_openai(prompt, model_name, temperature=0, top_p=1.0):
    response = client.responses.create(
        model=model_name,
        input=prompt,
        temperature=temperature,
        top_p=top_p
    )
    return response.output_text.strip()

def run_claude(prompt, model_name):
    response = claude_client.messages.create(
        model=model_name,
        max_tokens=50,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.content[0].text.strip()


def run_hf(prompt, model_name):
    completion = hf_client.chat.completions.create(
        model=model_name,
        temperature=0,
        top_p=1.0,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=50
    )
    return completion.choices[0].message.content.strip()

def get_questions(filename=None, data_dir=None, country=None, template='{country}_questions.csv'):
    if filename is None:
        filename = template.replace('{country}', country.replace(' ', '_'))

    if data_dir is None:
        raise ValueError('No data directory given')

    return pd.read_csv(os.path.join(data_dir, filename), encoding='utf-8-sig')

def get_annotations(filename=None, data_dir=None, country=None, template='{country}_data.json'):
    if filename is None:
        filename = template.replace('{country}', country.replace(' ', '_'))

    if data_dir is None:
        raise ValueError('No data directory given')

    with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
        return json.load(f)

def write_csv_row(row, filepath):
    file_exists = os.path.exists(filepath)

    with open(filepath, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def get_json_str(response):
    try:
        response = str(response).replace('\n', ' ')
        response = response.replace('```json', '').replace('```', '').strip()

        match = re.findall(r'\{.*\}', response)
        if not match:
            return response

        return json.loads(match[-1])
    except:
        return response

def is_float(x):
    try:
        float(str(x).strip())
        return True
    except:
        return False

def is_time_format(x):
    try:
        datetime.strptime(str(x).strip(), "%H:%M")
        return True
    except:
        return False

def is_date_format(x):
    text = str(x).strip()
    for fmt in ("%m/%d", "%m-%d", "%d/%m", "%d-%m"):
        try:
            datetime.strptime(text, fmt)
            return True
        except:
            continue
    return False