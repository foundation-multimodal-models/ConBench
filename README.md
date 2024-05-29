# Unveiling the Tapestry of Consistency in Large Vision-Language Models

:fire: Official implementation of paper: [**Unveiling the Tapestry of Consistency in Large Vision-Language Models**](https://arxiv.org/pdf/2405.14156)

By [Yuan Zhang](https://gumpest.github.io/), Fei Xiao, [Tao Huang](https://taohuang.info/), Chun-Kai Fan, Hongyuan Dong, Jiawen Li, Jiacong Wang, [Kuan Cheng](https://cfcs.pku.edu.cn/people/faculty/kuancheng/index.htm), [Shanghang Zhang](https://idm.pku.edu.cn/info/1017/1598.htm), [Haoyuan Guo](https://scholar.google.com/citations?user=hql67boAAAAJ&hl=en)

<p align='center'>
<img src='./assests/ConBench.png' alt='mask' width='400px'>
</p>

## Preparation

### Install ANLS


```shell
pip install anls
```

### Install OpenAI API

> [!Note]
> If you want to compute Score[C], you should install openAI API.

```shell
pip install openai
```

### ConBench Dataset

Download on [Huggingface](https://huggingface.co/datasets/ConBench/ConBench)

## Usage

### Get Model Responses

The model outputs answers based on the image and propmpt and stores them in format of txt.

Example for evaluating GPT-4V result in  [GPT-4V.py](https://github.com/open-mmlab/mmrazor)

Example for evaluating LLaVA-NEXT-34B result in  [LLaVA-NEXT-34B.py](https://github.com/open-mmlab/mmrazor)

The results should be listed like this:

```shell
GPT-4V
├─artwork.txt
├─attribute_reason.txt
├─attribute_recognition.txt
├─biology.txt
├─calculation.txt
├─celebrity.txt
├─chemistry.txt
├─code.txt
├─color.txt
├─count.txt
├─cross_instance_reason.txt
├─landmark.txt
├─math.txt
├─ocr.txt
├─physics.txt
├─position.txt
├─poster.txt
├─scene.txt
└translation.txt
```

### Fast Evaluation on ConScore[D]


```shell
python Score.py --results_dir ${Model_results} --Score_D
```

Example for evaluating GPT-4V result:

```shell
python3 Score.py --results_dir ./Res/GPT-4V --Score_D
```

The results will be save in `Con_res/GPT-4V_D.json`.

### Evaluation on ConScore[C]

```shell
python Score.py --results_dir ${Model_results} 
```

Example for evaluating GPT-4V result:

```shell
python3 Score.py --results_dir ./Res/GPT-4V
```

The results will be save in `Con_res/GPT-4V_C.json`.


## Leaderboard

### ConScore[D]

|        Rank         |         Teacher         | ConScore[D] |
| :--------------------: | :---------------------: | :------: |
| 1 | Qwen-VL-Max |  37.00   |
|  2  |  GPT-4-Omni  |   35.70   |
|  3  |    InternVL-v1.2P-40B    |   34.70   |
|  4  |    Gemini-Ultra-Vision    |   33.10   |
|  5  |    InternVL-v1.5-26B    |   31.40   |

### ConScore[C]

|        Rank         |         Teacher         | ConScore[C] |
| :--------------------: | :---------------------: | :------: |
| 1 | GPT-4-Omni  |  62.2   |
|  2  |  Qwen-VL-Max |   58.4   |
|  3  |    GPT-4V   |   55.6   |
|  4  |    Gemini-Ultra-Vision    |   54.6   |
|  5  |    InternVL-v1.2P-40B   |   53.7   |

* The review pipeline


<p align='center'>
<img src='./assests/pipeline.png' alt='mask' width='720px'>
</p>

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

If you use ConBench in your research, please cite our work by using the following BibTeX entry:
```bibtex
@article{zhang2024unveiling,
  title={Unveiling the Tapestry of Consistency in Large Vision-Language Models},
  author={Zhang, Yuan and Xiao, Fei and Huang, Tao and Fan, Chun-Kai and Dong, Hongyuan and Li, Jiawen and Wang, Jiacong and Cheng, Kuan and Zhang, Shanghang and Guo, Haoyuan},
  journal={arXiv preprint arXiv:2405.14156},
  year={2024}
}
```
