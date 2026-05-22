import pandas as pd
from openai import OpenAI
from huggingface_hub import InferenceClient
import anthropic
import os
import time



folders = "questions"


OPENAI_API_KEY = "KEY"
client = OpenAI(api_key=OPENAI_API_KEY)


CLAUDE_API_KEY = "KEY"
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

HF_TOKEN = "KEY"

hf_client = InferenceClient(provider="auto",api_key=HF_TOKEN)



models = [
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


def prompt_maker(question_file, region_en, region_ar):
    #Egyptian dialect
    templates = {
        "inst-4_eng": "Read the following question and provide a single answer without any explanations. Question: {q} Answer:",
        "pers-3_eng": "You are a person from {r} who is trying to explain your region's culture to a foreigner. Answer the following question, provide exactly a single answer without any explanations. {q}",
        "inst-4_ar": "اقرأ السؤال اللي جاي وادّي إجابة واحدة من غير أي شرح. السؤال: {q} الإجابة:",
        "pers-3_ar": "إنت من {r} وبتحاول تشرح ثقافة منطقتك لواحد أجنبي. جاوب على السؤال اللي جاي بإجابة واحدة من غير أي شرح. {q}"
    }
    #Modern Standard Arabic
    # templates = {
    #    "inst-4_eng": "Read the following question and provide a single answer without any explanations. Question: {q} Answer:",
    #    "pers-3_eng": "You are a person from {r} who is trying to explain your region's culture to a foreigner. Answer the following question and provide exactly one answer without any explanations. {q}",
    #    "inst-4_ar": "اقرأ السؤال التالي وقدّم إجابة واحدة فقط دون أي توضيحات. السؤال: {q} الإجابة:",
    #    "pers-3_ar": "أنت شخص من {r} وتحاول شرح ثقافة منطقتك لشخص أجنبي. أجب عن السؤال التالي، وقدّم إجابة واحدة فقط دون أي توضيحات. {q}",
    # }

    question_path = os.path.join(folders, question_file)
    df = pd.read_csv(question_path, encoding="utf-8-sig")

    prompt_list_inst_4_eng = []
    prompt_list_pers_3_eng = []
    prompt_list_inst_4_ar = []
    prompt_list_pers_3_ar = []


    if "Translation" in df.columns:
        for _, row in df.iterrows():
            id_ = row["ID"]
            translation = row["Translation"]

            prompt_list_pers_3_eng.append({
                "ID": id_,
                "Translation": translation,
                "prompt": templates["pers-3_eng"].format(r=region_en, q=translation),
                "response": "",
                "prompt_no": "pers-3",
            })

            prompt_list_inst_4_eng.append({
                "ID": id_,
                "Translation": translation,
                "prompt": templates["inst-4_eng"].format(q=translation),
                "response": "",
                "prompt_no": "inst-4",
            })

    if "Question" in df.columns:
        for _, row in df.iterrows():
            id_ = row["ID"]
            question = row["Question"]

            prompt_list_pers_3_ar.append({
                "ID": id_,
                "Translation": question,
                "prompt": templates["pers-3_ar"].format(r=region_ar, q=question),
                "response": "",
                "prompt_no": "pers-3",
            })

            prompt_list_inst_4_ar.append({
                "ID": id_,
                "Translation": question,
                "prompt": templates["inst-4_ar"].format(q=question),
                "response": "",
                "prompt_no": "inst-4",
            })

    return prompt_list_pers_3_eng, prompt_list_pers_3_ar, prompt_list_inst_4_eng, prompt_list_inst_4_ar


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
            {"role": "user", "content": prompt}]
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

def run_model(prompt, model_name):
    if model_name == "gpt-5.4-mini":
        return run_openai(prompt, model_name)
    elif model_name == "gpt-5.4":
        return run_openai(prompt, model_name)
    elif model_name == "claude-haiku-4-5":
        return run_claude(prompt, model_name)
    elif model_name == "claude-sonnet-4-6":
        return run_claude(prompt, model_name)
    elif model_name == "claude-opus-4-6":
        return run_claude(prompt, model_name)
    elif model_name == "Qwen/Qwen2.5-7B-Instruct":
        return run_hf(prompt, model_name)
    elif model_name == "Qwen/Qwen2.5-14B-Instruct":
        return run_hf(prompt, model_name)
    elif model_name == "Qwen/Qwen2.5-32B-Instruct":
        return run_hf(prompt, model_name)
    elif model_name == "Qwen/Qwen2.5-72B-Instruct":
        return run_hf(prompt, model_name)
    elif model_name == "CohereLabs/aya-expanse-32b":
        return run_hf(prompt, model_name)
    elif model_name == "meta-llama/Llama-3.1-8B-Instruct":
        return run_hf(prompt, model_name)
    elif model_name == "meta-llama/Llama-3.3-70B-Instruct":
        return run_hf(prompt, model_name)
    elif model_name == "CohereLabs/c4ai-command-r7b-arabic-02-2025":
        return run_hf(prompt, model_name)
    elif model_name == "silma-ai/SILMA-9B-Instruct-v1.0":
        return run_hf(prompt, model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def csv_maker(region_name, model_name):
    safe_name = model_name.replace("/", "_")

    pers_3_english, pers_3_arabic, inst_4_english, inst_4_arabic = prompt_maker(
        "",  #fill in  the name of region as used in the question file
        "", #fill in the the region name in English
        "" #fill in the region name in Arabic
    )


    files_to_save = [
        (pers_3_english, "English", "pers-3"),
        (pers_3_arabic, "Arabic", "pers-3"),
        (inst_4_english, "English", "inst-4"),
        (inst_4_arabic, "Arabic", "inst-4"),
    ]

    output_folder = os.path.join("model_inference_results", region_name)
    os.makedirs(output_folder, exist_ok=True)

    for prompt_list, language, prompt_no in files_to_save:
        filename = f"{safe_name}_{region_name}_{language}_{prompt_no}_result.csv"
        full_path = os.path.join(output_folder, filename)
        df_result = pd.DataFrame(prompt_list)
        df_result.to_csv(full_path, index=False, encoding="utf-8-sig")


def fill_csv_responses(region_name, model_name, n=200):
    safe_name = model_name.replace("/", "_")
    files = [
        f"{safe_name}_{region_name}_English_pers-3_result.csv",
        f"{safe_name}_{region_name}_Arabic_pers-3_result.csv",
        f"{safe_name}_{region_name}_English_inst-4_result.csv",
        f"{safe_name}_{region_name}_Arabic_inst-4_result.csv",
    ]

    output_folder = os.path.join("model_inference_results", region_name)

    for filename in files:
        full_path = os.path.join(output_folder, filename)
        df = pd.read_csv(full_path)

        if "response" not in df.columns:
            df["response"] = ""
        else:
            df["response"] = df["response"].fillna("").astype(str)

        df_subset = df.iloc[:n]

        empty_indices = df_subset.index[df_subset["response"].str.strip() == ""].tolist()

        for i in empty_indices:
            try:
                answer = run_model(df.at[i, "prompt"], model_name)
                df.at[i, "response"] = answer
                df.to_csv(full_path, index=False, encoding="utf-8-sig")
                time.sleep(1)

            except Exception as e:
                print(f"Error on {filename}, row {i + 1}: {e}")

                df.to_csv(full_path, index=False, encoding="utf-8-sig")
                print(f"Progress saved before continuing: {full_path}")
                continue


# run the code
model_name = ""#fill in the model name you want to run
csv_maker("", model_name)#fill in the question file
fill_csv_responses("", model_name)#fill in the region name in English

