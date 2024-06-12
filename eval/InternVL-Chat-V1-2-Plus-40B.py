import torch
from PIL import Image

import os
import json
from transformers import AutoModel, CLIPImageProcessor
from transformers import AutoTokenizer


if __name__ == '__main__':

    category_list = ["artwork", "attribute_reason", "attribute_recognition", "biology", \
        "calculation", "celebrity", "chemistry", "code", "color", "count", "cross_instance_reason", \
        "landmark", "math", "ocr", "physics", "position", "poster", "scene", "translation"]

    path = "OpenGVLab/InternVL-Chat-V1-2-Plus"
    # If you have an 80G A100 GPU, you can put the entire model on a single GPU.
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()

    # set the max number of tiles in `max_num`

    generation_config = dict(
        num_beams=1,
        max_new_tokens=512,
        do_sample=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(path)


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
            image = Image.open(img_file).convert('RGB')
            image = image.resize((448, 448))
            image_processor = CLIPImageProcessor.from_pretrained(path)

            pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            # try:
            res = model.chat(tokenizer, pixel_values, qs, generation_config)
            # except:
                # res = "None"
            print(res)
            print("-"*50)

            record = "%s"%info['image'] + "\t" + "%s"%info['question'] + "\t" + "%s"%info['answer'] + "\t" + "%s"%res.replace("\n", "") + "\n"
            pred_answer_list.append(record)

        with open("./Res/InternVL-Chat-V1-2-Plus/%s.txt"%category, "w") as file:
            file.writelines(pred_answer_list)