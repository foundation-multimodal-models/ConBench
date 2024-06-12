import base64
import requests

import openai
import pandas as pd
import base64
import os
import time
import json

# OpenAI API Key
api_key = "your-key"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def testOpenaiChatCompletions(image_path, qs):
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": qs
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response

if __name__ == '__main__':

    category_list = ["artwork", "attribute_reason", "attribute_recognition", "biology", \
        "calculation", "celebrity", "chemistry", "code", "color", "count", "cross_instance_reason", \
        "landmark", "math", "ocr", "physics", "position", "poster", "scene", "translation"]

    for category in category_list:
        print("Eval on " + category)

        testdata_root = "Conbench/%s"%category
        json_file = os.path.join(testdata_root, 'output_w_caption_%s.json'%category)
        infos = json.load(open(json_file, 'r'))

        pred_answer_list = []
        for info in infos:

            if info["question_field"] == "Choices":
                qs = info['question'] + " Please output like letters) corresponding text."
            elif info["question_field"] == "Caption":
                qs = info['question'] + " Please output only in a paragraph."
            else:
                qs = info['question']
            img_file = os.path.join(testdata_root, info['image'])
            try:
                resp = testOpenaiChatCompletions(img_file, qs)
                res = resp.json()['choices'][0]['message']['content']
            except:
                res = "None"
            print(res)
            print("-"*50)
            # time.sleep(2)

            record = "%s"%info['image'] + "\t" + "%s"%info['question'] + "\t" + "%s"%info['answer'] + "\t" + "%s"%res.replace("\n", "") + "\n"
            pred_answer_list.append(record)

        with open("./Res/GPT-4o/%s.txt"%category, "w") as file:
            file.writelines(pred_answer_list)