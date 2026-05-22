import os
import re
import json
import csv
import argparse
import unicodedata as ud
from string import punctuation

import pandas as pd
from tqdm.auto import tqdm
import spacy
from qalsadi.lemmatizer import Lemmatizer as ARLeammatizer

ALL_MODELS = [
    "gpt-5.4-mini",
    "gpt-5.4",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-opus-4-6",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "CohereLabs/aya-expanse-32b",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "CohereLabs/c4ai-command-r7b-arabic-02-2025",
    "silma-ai/SILMA-9B-Instruct-v1.0",
]


def write_csv_row(values, filename):
    print("Writing row to", filename, ":", values)
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(values)


def get_annotations(data_dir, country, template="{country}_data.json"):
    filename = template.replace("{country}", country.replace(" ", "_"))
    with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_response_file(
    filename=None,
    data_dir=None,
    model=None,
    country=None,
    language=None,
    prompt_no=None,
    template="{model}_{region_name}_{language}_{prompt_no}_result.csv"
):
    if filename is None:
        safe_model = model.replace("/", "_")
        filename = (
            template.replace("{model}", safe_model)
            .replace("{region_name}", country.replace(" ", "_"))
            .replace("{language}", language)
            .replace("{prompt_no}", prompt_no)
        )
        print("Filename:", filename)

    if data_dir is None:
        raise ValueError("ERROR: No data directory given")

    full_path = os.path.join(data_dir, country, filename)
    print("Reading:", full_path)

    model_res_df = pd.read_csv(full_path, encoding="utf-8")
    return model_res_df


def delete_prompt_from_answer(text, prompt):
    text = str(text).replace(prompt, "").replace("：", ":").replace("、", ",").replace("，", ",").replace("。", ".").lower()
    prompt = str(prompt).replace("：", ":").replace("、", ",").replace("，", ",").replace("。", ".").lower()

    match = re.findall(r"^(\w+:)\s", text)
    extracted = ""
    for m in match:
        if len(m) > len(extracted) and m.replace(":", "") in prompt:
            extracted = m

    if match:
        return text.replace(extracted, "").strip()
    return text.strip()


def get_llm_response_by_id(res_df, qid, id_col, r_col):
    qid = str(qid).strip()
    res_df = res_df.copy()
    res_df[id_col] = res_df[id_col].astype(str).str.strip()

    if qid not in set(res_df[id_col]):
        print(qid, "not in LLM response df")
        return None

    try:
        matched_rows = res_df[res_df[id_col] == qid]
        llm_response = matched_rows[r_col].values[-1]
        prompt = matched_rows["prompt"].values[-1]
        llm_response = delete_prompt_from_answer(llm_response, prompt)
        llm_response = llm_response.strip(".").lower()
        return llm_response
    except Exception:
        print(res_df[res_df[id_col] == qid])
        return None


def normalize_arabic(text):
    text = str(text).strip().lower()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و").replace("ئ", "ي")
    text = text.replace("ة", "ه")
    text = text.replace("ـ", "")
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_tokens(tokens):
    d = {ord("\N{COMBINING ACUTE ACCENT}"): None}
    return [
        ud.normalize("NFD", str(term)).translate(d).lower()
        for term in tokens
        if str(term).strip() != "" and str(term) not in punctuation
    ]

def lemma_check(answer, llm_response, en_lemmatizer, language):
    answer = "" if answer is None else str(answer).strip()
    llm_response = "" if llm_response is None else str(llm_response).strip()

    if answer == "" or llm_response == "":
        return False

    if (
        answer.lower() in llm_response.lower()
        or answer.replace("-", " ").lower() in llm_response.lower()
        or answer.replace(" ", "-").lower() in llm_response.lower()
    ):
        return True

    if language == "Arabic":
        lemmatizer = ARLeammatizer()

        answer = normalize_arabic(answer)
        llm_response = normalize_arabic(llm_response)

        if (
            answer in llm_response
            or answer.replace("-", " ") in llm_response
            or answer.replace(" ", "-") in llm_response
        ):
            return True

        answer_tokens = lemmatizer.lemmatize(answer)
        llm_tokens = lemmatizer.lemmatize(llm_response)

        if isinstance(answer_tokens, str):
            answer_tokens = answer_tokens.split()
        if isinstance(llm_tokens, str):
            llm_tokens = llm_tokens.split()

    elif language == "English":
        answer_tokens = [token.lemma_ for token in en_lemmatizer(answer)]
        llm_tokens = [token.lemma_ for token in en_lemmatizer(llm_response)]

    else:
        answer_tokens = answer.split()
        llm_tokens = llm_response.split()

    answer_tokens = clean_tokens(answer_tokens)
    llm_tokens = clean_tokens(llm_tokens)

    for a in answer_tokens:
        if a not in llm_tokens:
            return False
    return True

def soft_exact_match_fixed(country, language, annotation_dict, response_df, id_col, r_col, annotations_key="annotations"):
    binary_score = 0
    weight_score = 0
    valid_question_cnt = 0

    en_lemmatizer = spacy.load("en_core_web_sm")

    response_df = response_df.copy()
    response_df[id_col] = response_df[id_col].astype(str).str.strip()
    response_df["binary_score"] = None
    response_df["weight_score"] = None

    pb = tqdm(annotation_dict.items(), total=len(annotation_dict))

    for qid, data in pb:
        qid = str(qid).strip()
        pb.set_description(qid)

        if (
            data["idks"]["no-answer"] + data["idks"]["not-applicable"] >= 3
            or data["idks"]["idk"] >= 5
            or len(data[annotations_key]) == 0
        ):
            continue

        valid_question_cnt += 1
        llm_response = get_llm_response_by_id(response_df, qid, id_col, r_col)
        flag = False
        matched_weight = 0.0

        if llm_response and data[annotations_key]:
            max_vote = data[annotations_key][0]['count']

            for agg_ans in data[annotations_key]:
                if language != "English":
                    for a in agg_ans["answers"]:
                        if lemma_check(a, llm_response, en_lemmatizer, "Arabic"):
                            binary_score += 1
                            matched_weight = agg_ans["count"] / max_vote
                            weight_score += matched_weight
                            flag = True
                            break

                if not flag:
                    for a in agg_ans["en_answers"]:
                        if lemma_check(a, llm_response, en_lemmatizer, "English"):
                            binary_score += 1
                            matched_weight = agg_ans["count"] / max_vote
                            weight_score += matched_weight
                            flag = True
                            break

                if flag:
                    break

        if flag:
            response_df.loc[response_df[id_col] == qid, "binary_score"] = 1
            response_df.loc[response_df[id_col] == qid, "weight_score"] = matched_weight
        else:
            response_df.loc[response_df[id_col] == qid, "binary_score"] = 0
            response_df.loc[response_df[id_col] == qid, "weight_score"] = 0

        pb.set_postfix({
            "bs": binary_score / valid_question_cnt * 100,
            "ws": weight_score / valid_question_cnt * 100
        })

    binary_score = binary_score / valid_question_cnt * 100
    weight_score = weight_score / valid_question_cnt * 100

    print("SEM-B:", binary_score)
    print("SEM-W:", weight_score)

    return binary_score, weight_score, response_df


def evaluate_all_models_for_region(
    country,
    response_dir,
    annotation_dir,
    id_col="ID",
    response_col="response",
    eval_res_filename="evaluation_results.csv",
    annotation_template="{country}_data.json"
):
    combinations = [
        ("English", "pers-3"),
        ("Arabic", "pers-3"),
        ("English", "inst-4"),
        ("Arabic", "inst-4"),
    ]

    for model in ALL_MODELS:
        for language, prompt_no in combinations:
            try:
                evaluate_one(
                    model=model,
                    country=country,
                    language=language,
                    prompt_no=prompt_no,
                    response_dir=response_dir,
                    annotation_dir=annotation_dir,
                    id_col=id_col,
                    response_col=response_col,
                    eval_res_filename=eval_res_filename,
                    annotation_template=annotation_template
                )
            except Exception as e:
                print(f"Skipping {model} | {country} | {language} | {prompt_no} because of error: {e}")


def evaluate_one(
    model,
    country,
    language,
    prompt_no,
    response_dir,
    annotation_dir,
    id_col,
    response_col,
    eval_res_filename,
    annotation_template="{country}_data.json"
):
    eval_output_folder = os.path.join("evaluation_results", country)
    os.makedirs(eval_output_folder, exist_ok=True)
    eval_res_path = os.path.join(eval_output_folder, eval_res_filename)

    if not os.path.exists(eval_res_path):
        write_csv_row(
            ["model_name", "region_name", "language", "prompt_no", "SEM-B", "SEM-W"],
            eval_res_path
        )

    res_df = get_model_response_file(
        data_dir=response_dir,
        model=model,
        country=country,
        language=language,
        prompt_no=prompt_no
    )

    annotation_dict = get_annotations(
        data_dir=annotation_dir,
        country=country,
        template=annotation_template
    )

    sem_b, sem_w, scored_df = soft_exact_match_fixed(
        country=country,
        language=language,
        annotation_dict=annotation_dict,
        response_df=res_df,
        id_col=id_col,
        r_col=response_col,
        annotations_key="annotations"
    )

    write_csv_row(
        [model, country, language, prompt_no, round(sem_b, 2), round(sem_w, 2)],
        eval_res_path
    )

    score_output_folder = os.path.join("model_inference_result_scores", country)
    os.makedirs(score_output_folder, exist_ok=True)

    safe_model = model.replace("/", "_")
    score_filename = f"{safe_model}_{country}_{language}_{prompt_no}_response_score.csv"
    score_path = os.path.join(score_output_folder, score_filename)

    scored_df.to_csv(score_path, index=False, encoding="utf-8-sig")

    df = pd.read_csv(eval_res_path)
    df.drop_duplicates(
        subset=["model_name", "region_name", "language", "prompt_no"],
        keep="last",
        inplace=True
    )
    df.to_csv(eval_res_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--country", type=str, required=True)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--prompt_no", type=str, default=None)

    parser.add_argument("--response_dir", type=str, default="model_inference_results")
    parser.add_argument("--annotation_dir", type=str, default="annotations")
    parser.add_argument("--id_col", type=str, default="ID")
    parser.add_argument("--response_col", type=str, default="response")
    parser.add_argument("--evaluation_result_file", type=str, default="evaluation_results.csv")
    parser.add_argument("--annotation_filename", type=str, default="{country}_data.json")

    parser.add_argument("--all_models_region", action="store_true")

    args = parser.parse_args()

    if args.all_models_region:
        evaluate_all_models_for_region(
            country=args.country,
            response_dir=args.response_dir,
            annotation_dir=args.annotation_dir,
            id_col=args.id_col,
            response_col=args.response_col,
            eval_res_filename=args.evaluation_result_file,
            annotation_template=args.annotation_filename
        )
    else:
        if args.model is None or args.language is None or args.prompt_no is None:
            raise ValueError("For single evaluation, you need --model, --language, and --prompt_no.")

        evaluate_one(
            model=args.model,
            country=args.country,
            language=args.language,
            prompt_no=args.prompt_no,
            response_dir=args.response_dir,
            annotation_dir=args.annotation_dir,
            id_col=args.id_col,
            response_col=args.response_col,
            eval_res_filename=args.evaluation_result_file,
            annotation_template=args.annotation_filename
        )