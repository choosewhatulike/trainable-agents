# Character-LLM: A Trainable Agent for Role-Playing

<p align="center">
<a href="https://github.com/choosewhatulike/character-llm/blob/main/LICENSE">
<img src='https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg'></a>
<img src='https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg'>
<img src='https://img.shields.io/badge/python-3.8+-blue.svg'>
</p>

<p align="center">
🤗 <a href="" target="_blank">Models (Upcoming)</a> • 🤗 <a href="" target="_blank">Dataset (Upcoming)</a> • 📃 <a href="https://arxiv.org/abs/2310.10158" target="_blank">Character-LLM</a><br>
</p>

This is the official repository of our [EMNLP 2023 paper](https://arxiv.org/abs/2310.10158). Welcome! 🤩🤩🤩

We introduce **Character-LLMs** a trainable agent for role-playing that learns from actual experiences, characteristics, and emotions. Compared with prompted agents, Character-LLMs are trainable agents that specifically trained for role-playing, which are able to act as specific people, such as Beethoven, Queen Cleopatra, Julius Caesar, etc, with detailed character-related knowledge and representative character personalities. No additional prompt or reference document is needed. To achieve this, we propose **Experience Reconstruction**, a data generation process that can generates detailed and diverse experience data of certain character for training. For more details, please refer to the [paper](https://arxiv.org/abs/2310.10158).

<p align="center">
    Overview of the construction flow of Character-LLM.
    <img src="./images/method1.png" width="100%"> <br>
    <br>
</p>

## Generated Samples Demonstration 📝

<p align="center">
    Single-turn interview outputs from different methods simulating Beethoven.
    <img src="./images/result1.png" width="95%"> <br>
    <br>
</p>

<p align="center">
    Multi-turn interview outputs from our trainable agent of Cleopatra VII.
    <img src="./images/result2.png" width="95%"> <br>
    <br>
</p>

<p align="center">
    Multi-turn interview outputs from our trainable agent of Socrates.
    <img src="./images/result3.png" width="95%"> <br>
    <br>
</p>

## Dataset & Model Weights 📚
As we are working on camera ready right now, datasets and model weights are going to be released in the near future. Stay tuned! 🤩🤩🤩

## Citation 📖

Please cite our work if you found the resources in this repository useful:
```bib
@inproceedings{shao2023character,
    title={Character-LLM: A Trainable Agent for Role-Playing},
    author={Yunfan Shao and Linyang Li and Junqi Dai and Xipeng Qiu},
    booktitle={EMNLP},
    year=2023
}
```

## Acknowledgements 🥰
- We especially thank Ming Zhong for the helpful proofreading and suggestions on the paper.
- This work was supported by the National Key Research and Development Program of China (No.2022ZD0160102) and National Natural Science Foundation of China (No.62022027). 


## Limitations ❗
The resources, including generated data, code and models, associated with this project are restricted for academic research purposes only and cannot be used for commercial purposes. The contents produced by Character-LLMs are influenced by uncontrollable variables such as randomness, and therefore, the accuracy and quality of the output cannot be guaranteed by this project. The authors of this project are not responsible for any potential consequences caused by the use of the resources in this project. 
