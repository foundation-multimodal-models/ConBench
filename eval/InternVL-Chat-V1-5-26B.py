from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image

from torchvision.transforms.functional import InterpolationMode

import os
import time
import json


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


if __name__ == '__main__':

    category_list = ["artwork", "attribute_reason", "attribute_recognition", "biology", \
        "calculation", "celebrity", "chemistry", "code", "color", "count", "cross_instance_reason", \
        "landmark", "math", "ocr", "physics", "position", "poster", "scene", "translation"]

    path = "OpenGVLab/InternVL-Chat-V1-5"
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

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

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
            pixel_values = load_image(img_file, max_num=6).to(torch.bfloat16).cuda()

            # try:
            res = model.chat(tokenizer, pixel_values, qs, generation_config)
            # except:
                # res = "None"
            print("-"*50)

            record = "%s"%info['image'] + "\t" + "%s"%info['question'] + "\t" + "%s"%info['answer'] + "\t" + "%s"%res.replace("\n", "") + "\n"
            pred_answer_list.append(record)

        with open("./Res/InternVL-Chat-V1-5/%s.txt"%category, "w") as file:
            file.writelines(pred_answer_list)