import openai
import base64
import os
import time
import json

def testOpenaiChatCompletions(img, content):
    client = openai.AzureOpenAI(
        azure_endpoint="azure_endpoint",
        api_version="api_version",
        api_key="api_key"
    )
    completion = client.chat.completions.create(
        model="gptv",
        messages=[
            {
                "role": "user",
                "content": [content, {"image": img}]
            }
        ]
    )
    return completion


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

            img_file = os.path.join(testdata_root, info['image'])
            image = open(img_file, "rb")
            b64string = base64.b64encode(image.read()).decode("utf-8")
            try:
                resp = testOpenaiChatCompletions(b64string, qs)
                res = resp.choices[0].message.content
            except:
                res = "None"
            print(res)
            print("-"*50)
            time.sleep(1)

            record = "%s"%info['image'] + "\t" + "%s"%info['question'] + "\t" + "%s"%info['answer'] + "\t" + "%s"%res.replace("\n", "") + "\n"
            pred_answer_list.append(record)

        with open("./Res/GPT-4/%s.txt"%category, "w") as file:
            file.writelines(pred_answer_list)