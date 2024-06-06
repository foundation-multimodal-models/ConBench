'''
Copyright (2024) Bytedance Ltd. 

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
'''

import os
import argparse
from anls import anls_score

import openai
import time
import json


parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', default='./LaVIN', type=str)
parser.add_argument("--Score_D", action="store_true", help="utilize GPT or not")
parser.add_argument("--save_dir", default='./Con_res', type=str)


# 19 classes
eval_type_dict = {
    "Sensation": ["count","color", "scene", "poster", "attribute_recognition", "ocr", "position"],
    "Cognition": ["calculation", "code", "translation", "math", "cross_instance_reason", "attribute_reason"],
    "Knowledge": ["celebrity", "chemistry", "physics", "biology", "landmark", "artwork"]
}

def testOpenaiChatCompletions(query):
    client = openai.AzureOpenAI(
        azure_endpoint="your_azure_endpoint",
        api_version="your_api_version",
        api_key="your_api_key"
    )
    
    completion = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ]
    )
    return completion


def GPTAnswer(Cap_ans, Ques, pred_ans):

    query = "Now, you are a teaching expert. I will provide you with a passage, a question, and an answer filled in by students. \npassage: {}. \nquestions: {} \nstudents answers: {} \nPlease help me fairly determine whether the student answer is correct with the answer you provided. Only output yes or no.".format(Cap_ans, Ques, pred_ans)

    try:
        resp = testOpenaiChatCompletions(query)
    except:
        resp = "None"
    print("Con Res: " + resp.choices[0].message.content)

    time.sleep(1)
    return resp.choices[0].message.content


class calculate_metrics:
    def divide_chunks(self, l, n=4):
        for i in range(0, len(l), n): 
            yield l[i:i + n]
        
        return 

    def parse_pred_ans_NY(self, pred_ans):
        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        else:
            prefix_pred_ans = pred_ans[:4]

            if "yes" in prefix_pred_ans:
                pred_label = "yes"
            elif "no" in prefix_pred_ans:
                pred_label = "no"
            else:
                pred_label = "other"

        return pred_label

    def parse_pred_ans_choice(self, pred_ans):

        return pred_ans.replace(" ", "")[0]


    def process_result(self, results_dir, save_dir):
        model_name = results_dir.split('/')[-1]
        question_dict = {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }
        total_img_num = 0

        task_score_dict = dict()

        con_dict = {
            "choice_qa": 0,
            "choice_qa_false": 0,
        }
        # 1
        con_Cap_dict = {
            0: 0,
            1: 0,
            2: 0,
        }

        for eval_type, task_name_list in eval_type_dict.items():
            print("===========", eval_type, "===========")

            for task_name in task_name_list:
                question_dict_sub = {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0 # all cor
                }
                con_dict_sub = {
                    "choice_qa": 0,
                    "choice_qa_false": 0,
                }
                con_Cap_dict_sub = {
                    0: 0,
                    1: 0,
                    2: 0,
                }

                task_txt = os.path.join(results_dir, task_name + ".txt")
                lines = open(task_txt, 'r').readlines()
                chunk_lines = list(self.divide_chunks(lines)) # one image corresponds to three questions
                
                img_num = len(chunk_lines)
                total_img_num += img_num

                for img_idx, img_items in enumerate(chunk_lines):
                    # Each image with 4 different QAs
                    assert len(img_items) == 4

                    # extract each image's Caption
                    if args.Score_D:
                        print(img_items[-1])
                        pass
                    _, _, _, pred_ans = img_items[-1].split("\t")
                    Cap_ans = pred_ans.replace('\n', "")

                    # 3 qas all correct
                    img_3_cor_num = 0
                    # save true or false
                    ans_dict = {
                        0: 0,
                        1: 0,
                        2: 0,
                    }
                    # save answer
                    ans_memory_dict = {
                        0: "", # ny
                        1: "", # choice
                        2: "", # qa
                    }

                    for qa_idx, img_item in enumerate(img_items[:-1]):
                        # Each question
                        img_name, question, gt_ans, pred_ans = img_item.split("\t")

                        # preprocess
                        gt_ans = gt_ans.lower()
                        pred_ans = pred_ans.replace('\n', "").lower()

                        if qa_idx == 0:
                            parse_pred_ans = self.parse_pred_ans_NY(pred_ans)
                        elif qa_idx == 1:
                            parse_pred_ans = self.parse_pred_ans_choice(pred_ans)
                        else:
                            parse_pred_ans = pred_ans

                        # save 3 kinds preds in each image
                        ans_memory_dict[qa_idx] = parse_pred_ans

                        if args.Score_D:
                            print(img_name, img_idx, gt_ans, parse_pred_ans)
                            print("-"*50)
                            GPT_out = "None"
                        else:
                            print("truth is ", gt_ans, " ||", "answer is ", pred_ans)
                            GPT_out = GPTAnswer(Cap_ans, question, pred_ans)
                        if self.parse_pred_ans_NY(GPT_out.lower()) == "yes":
                            con_Cap_dict[qa_idx] += 1
                            con_Cap_dict_sub[qa_idx] += 1
                        
                        # compute accuracy score
                        if (qa_idx == 2 and anls_score(prediction=parse_pred_ans, gold_labels=[gt_ans], threshold=0.95) >= 0.4) \
                            or (gt_ans == parse_pred_ans): 
                            # ANLS or string equal
                            img_3_cor_num += 1
                            question_dict[qa_idx] +=1
                            ans_dict[qa_idx] = 1
                            question_dict_sub[qa_idx] +=1

                    # 3 qas all correct
                    if img_3_cor_num == 3:
                        question_dict_sub[3] += 1
                        question_dict[3] += 1
                    
                    # --- Con between choice and qa ---
                    if (ans_dict[1] == ans_dict[2] == 1) or \
                        (ans_dict[1] == ans_dict[2] == 0 and anls_score(prediction=ans_memory_dict[1], gold_labels=[ans_memory_dict[2]], threshold=0.95) > 0.2):
                        con_dict["choice_qa"] += 1
                        con_dict_sub["choice_qa"] += 1
                    if (ans_dict[1] == ans_dict[2] == 0 and anls_score(prediction=ans_memory_dict[1], gold_labels=[ans_memory_dict[2]], threshold=0.95) > 0.2):
                        con_dict["choice_qa_false"] += 1
                        con_dict_sub["choice_qa_false"] += 1

                metric_dict = {}
                acc_NY = question_dict_sub[0] / img_num
                acc_Choice = question_dict_sub[1] / img_num
                acc_QA = question_dict_sub[2] / img_num
                acc_plus = question_dict_sub[3] / img_num
                con_qa = con_dict_sub["choice_qa"] / img_num
                con_qa_false = con_dict_sub["choice_qa_false"] / img_num
                con_Cap_ny = con_Cap_dict_sub[0] / img_num
                con_Cap_Choice = con_Cap_dict_sub[1] / img_num
                con_Cap_QA = con_Cap_dict_sub[2] / img_num  

                metric_dict["acc_NY"] = acc_NY
                metric_dict["acc_Choice"] = acc_Choice
                metric_dict["acc_QA"] = acc_QA
                metric_dict["ConScore[D]"] = acc_plus
                metric_dict["con_choice_qa"] = con_qa
                metric_dict["con_choice_qa_all_false"] = con_qa_false
                metric_dict["con_Cap_ny"] = con_Cap_ny
                metric_dict["con_Cap_Choice"] = con_Cap_Choice
                metric_dict["con_Cap_QA"] = con_Cap_QA
                task_score_dict[task_name] = metric_dict

        print(task_score_dict)

        # total computation
        print("total img num is", total_img_num)
        metric_dict = {}
        acc_NY = question_dict[0] / total_img_num
        acc_Choice = question_dict[1] / total_img_num
        acc_QA = question_dict[2] / total_img_num
        acc_plus = question_dict[3] / total_img_num
        con_qa = con_dict["choice_qa"] / total_img_num
        con_qa_false = con_dict["choice_qa_false"] / total_img_num
        
        con_Cap_ny = con_Cap_dict[0] / total_img_num
        con_Cap_Choice = con_Cap_dict[1] / total_img_num
        con_Cap_QA = con_Cap_dict[2] / total_img_num

        metric_dict["acc_NY"] = "{:.2f}%".format(acc_NY * 100)
        metric_dict["acc_Choice"] = "{:.2f}%".format(acc_Choice * 100)
        metric_dict["acc_QA"] = "{:.2f}%".format(acc_QA * 100)
        metric_dict["ConScore[D]"] = "{:.2f}%".format(acc_plus * 100)
        metric_dict["con_choice_qa"] = "{:.2f}%".format(con_qa * 100)
        metric_dict["con_choice_qa_all_false"] = "{:.2f}%".format(con_qa_false * 100)

        metric_dict["con_Cap_ny"] = "{:.2f}%".format(con_Cap_ny * 100)
        metric_dict["con_Cap_Choice"] = "{:.2f}%".format(con_Cap_Choice * 100)
        metric_dict["con_Cap_QA"] = "{:.2f}%".format(con_Cap_QA * 100)
        metric_dict["ConScore[C]"] = "{:.2f}%".format((con_Cap_ny + con_Cap_Choice + con_Cap_QA) / 3 * 100)

        print("total res:", metric_dict, "\n")

        # Core Computation
        core_score_dict = {}
        for eval_type, task_name_list in eval_type_dict.items():
            core_score_dict[eval_type] = {
                "acc_NY" : "{:.2f}%".format(sum([task_score_dict[task]["acc_NY"] for task in task_name_list]) / len(task_name_list) * 100),
                "acc_Choice" : "{:.2f}%".format(sum([task_score_dict[task]["acc_Choice"] for task in task_name_list]) / len(task_name_list) * 100),
                "acc_QA" : "{:.2f}%".format(sum([task_score_dict[task]["acc_QA"] for task in task_name_list]) / len(task_name_list) * 100),
                "ConScore[D]" : "{:.2f}%".format(sum([task_score_dict[task]["ConScore[D]"] for task in task_name_list]) / len(task_name_list) * 100),
                "con_choice_qa" : "{:.2f}%".format(sum([task_score_dict[task]["con_choice_qa"] for task in task_name_list]) / len(task_name_list) * 100),
                "con_choice_qa_all_false" : "{:.2f}%".format(sum([task_score_dict[task]["con_choice_qa_all_false"] for task in task_name_list]) / len(task_name_list) * 100),
                "con_Cap_ny" : "{:.2f}%".format(sum([task_score_dict[task]["con_Cap_ny"] for task in task_name_list]) / len(task_name_list) * 100),
                "con_Cap_Choice" : "{:.2f}%".format(sum([task_score_dict[task]["con_Cap_Choice"] for task in task_name_list]) / len(task_name_list) * 100),
                "con_Cap_QA" : "{:.2f}%".format(sum([task_score_dict[task]["con_Cap_QA"] for task in task_name_list]) / len(task_name_list) * 100),   
            }
        print(core_score_dict)
        # Save
        save_dict = {}
        save_dict["ConBench"] = "Copyright @ ByteDance."
        save_dict["Model"] = model_name
        save_dict["Summary"] = metric_dict
        save_dict["Core_Ability"] = core_score_dict
        save_dict["Detail_task"] = task_score_dict

        res_json = json.dumps(save_dict, indent=4)

        if not args.Score_D:
            with open(f'{save_dir}/{model_name}_C.json', 'w') as json_file:
                json_file.write(res_json)
        else:
            with open(f'{save_dir}/{model_name}_D.json', 'w') as json_file:
                json_file.write(res_json)  
        return 


if __name__ == "__main__":
    cal = calculate_metrics()

    args = parser.parse_args()
    results_dir = args.results_dir
    save_dir = args.save_dir

    cal.process_result(results_dir, save_dir)