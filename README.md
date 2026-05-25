# Beyond the Pyramids

## About

This repository contains the materials for **Beyond the Pyramids: Evaluating Regional and Linguistic Variation in LLMs' Knowledge of Egyptian Culture**, a Bachelor's thesis project in Artificial Intelligence at Vrije Universiteit Amsterdam.

This project adapts the BLEnD framework to the national and regional Egyptian contexts using newly collected datasets of everyday Egyptian cultural questions.

The datasets covers six Egyptian regions: Alexandria, Greater Cairo, Lower Egypt, Upper Egypt, Sinai, and the Suez Canal Cities. Both the question wording and the prompt language were adapted into three language varieties: English, Modern Standard Arabic (MSA), and the Egyptian dialect.
<p align="center">
  <img src="egypt-map.png" alt="Egypt map" width="350">
</p
The data was collected through a multi-phase Google Forms process. The questions were presented to participants in the Egyptian dialect. Participants had to have lived more than half of their lives in the region they were answering for. 

We evaluated how well Large Language Models answer cultural knowledge questions across these regions and language varieties. The evaluation was done in two parts: automated evaluation and human evaluation. The automated scoring was conducted across 14 models on both Short-Answer Questions(SAQ) and Multiple-Choice Questions(MCQ). Our evaluation used the the inst-4 and pers-3 prompts. Second, the zero-scored SAQ responses in The Egyptian dialect and Modern Standard Arabic(MSA) for GPT-5.4, Qwen2.5-72B, and command-r7b-arabic-02-2025 were manually assigned an applicability score to check whether the answers were still culturally or regionally possible. Responses that were not applicable at all were then categorized by the type of error the model made.

## Data

The human-annotated answers, can be found in the `data/annotations/` directory. This directory includes files for the six Egyptian regions and for Egypt as a whole. The Egypt-wide dataset is created by combining the answers from all the regional datasets.

Each region has two annotation files:

- `{region}_data.json` contains the Egyptian dialect version of the dataset.
- `{region}_MSA_data.json` contains the MSA version of the dataset.

The annotations are the same in both files, but the questions are written in different language varieties. Each file contains a JSON object where the unique question IDs are used as keys. The values include the question, the human annotations, and their vote counts.

The question files are located in the `data/questions/` directory. Each region has two question files:

- `{region}_questions.csv` contains the English and Egyptian dialect questions.
- `{region}_MSA_questions.csv` contains the English and Modern Standard Arabic questions.

The multiple-choice question files can be found in `evaluation/mc_data/{region}/`. Each region has three MCQ files:

- `mc_questions_{region}_ar.csv` for Egyptian dialect
- `mc_questions_{region}_en.csv` for English
- `mc_questions_{region}_msa.csv` for Modern Standard Arabic

The answer options are kept the same across the Egyptian dialect and MSA versions, with only the question wording changing. Each file includes the question, the answer options, and the correct answer.



## Code adaptation

The prompt templates are adapted into the in `model_inference.py`. The file contains two prompt settings: one for English and Egyptian dialect, and one for English and MSA. To run a specific setting, uncomment the prompt set you want to use and comment out the other one.

Our `evaluation_utils.py` combines the helper parts from BLEnD’s `utils.py` and `evaluation_utils.py`. From `utils.py`, it keeps the general idea of having helper functions for model access, JSON extraction, CSV writing, and simple format checks. However, it is simplified for this project by only supporting API-based inference through OpenAI, Anthropic, and Hugging Face. From BLEnD’s `evaluation_utils.py`, it keeps the functions for loading the question and annotation files.

The adapted `eval.py` combines the short-answer evaluation parts that were originally split across BLEnD’s `evaluate.py`, `evaluation_utils.py`, and `exact_match.py`. From `evaluate.py`, it keeps the role of running the evaluation and saving the final short-answer scores. From `evaluation_utils.py`, it keeps the logic for reading model response files, matching responses by question ID, and removing the prompt from the model answer if the model repeated it. From `exact_match.py`, it keeps the soft exact-match scoring logic, but this is now integrated directly into `eval.py` instead of being kept in a separate file.

For Arabic scoring, the adapted code also adds Arabic-specific normalization before Qalsadi lemmatization, which helps handle spelling differences and surface-form variation in MSA and Egyptian dialect more consistently.

For the MCQ evaluation, we used GPT-5.4 for the meaning-based similarity check between answer options and for generating dummy options when there were not enough valid regional distractors.

## Citation

This project adapts the BLEnD framework. If you use this repository, please also cite the original BLEnD paper.
