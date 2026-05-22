import os
import re
import json
import argparse
import pandas as pd
from tqdm import tqdm
import evaluation_utils as ev


def run_model(prompt, model_name, temperature=0, top_p=1.0):
    if model_name.startswith("gpt-"):
        return ev.run_openai(
            prompt,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p
        )
    elif model_name.startswith("claude-"):
        return ev.run_claude(
            prompt,
            model_name=model_name
        )
    else:
        return ev.run_hf(
            prompt,
            model_name=model_name
        )


def get_model_mc_response(
    model_name,
    mc_dir,
    questions_file,
    response_file=None,
    temperature=0,
    top_p=1.0
):
    if response_file is None:
        response_file = f"{model_name}_mc_res.csv"

    questions_path = os.path.join(mc_dir, questions_file)

    if os.path.dirname(response_file):
        response_path = response_file
    else:
        response_path = os.path.join(mc_dir, response_file)

    if os.path.dirname(response_path):
        os.makedirs(os.path.dirname(response_path), exist_ok=True)

    questions_df = pd.read_csv(questions_path, encoding='utf-8-sig')

    already = None
    expected_cols = list(questions_df.columns) + ['full_res', 'final_ans']

    if not os.path.exists(response_path):
        ev.write_csv_row(expected_cols, response_path)
    else:
        already = pd.read_csv(response_path, encoding='utf-8-sig')
        if list(already.columns) != expected_cols:
            print(f"Response file header mismatch. Recreating: {response_path}")
            os.remove(response_path)
            ev.write_csv_row(expected_cols, response_path)
            already = None

    done_ids = set(already['MCQID']) if isinstance(already, pd.DataFrame) else set()

    pb = tqdm(questions_df.iterrows(), total=len(questions_df))
    right = 0
    done = 0

    for _, row in pb:
        qid = row['MCQID']
        pb.set_description(str(qid))

        if qid in done_ids:
            done += 1
            continue

        prompt = row['prompt']
        print(prompt)

        full_res = run_model(
            prompt,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p
        )
        print(full_res)

        json_res = ev.get_json_str(full_res)

        if isinstance(json_res, dict) and 'answer_choice' in json_res:
            try:
                final_ans = re.findall(r'[A-D]', str(json_res['answer_choice']).upper())[0]

                if final_ans + '.' not in prompt:
                    for k, v in json.loads(row['choices']).items():
                        if v == json_res['answer_choice']:
                            final_ans = str(k)
                            break
                    else:
                        final_ans = full_res

            except Exception:
                for k, v in json.loads(row['choices']).items():
                    if v == json_res['answer_choice']:
                        final_ans = str(k)
                        break
                else:
                    final_ans = full_res
        else:
            try:
                final_ans = re.findall(r'[A-D]', str(json_res).upper())[0]
            except Exception:
                final_ans = full_res

        ev.write_csv_row(list(row) + [full_res, final_ans], response_path)

        done += 1
        if str(final_ans) == str(row['answer_idx']):
            right += 1

        pb.set_postfix({'score': right / max(done, 1)})


def multiple_choice_score(mc_dir, questions_file, mc_res_file):
    questions_path = os.path.join(mc_dir, questions_file)

    if os.path.dirname(mc_res_file):
        response_path = mc_res_file
    else:
        response_path = os.path.join(mc_dir, mc_res_file)

    qdf = pd.read_csv(questions_path, encoding='utf-8-sig')

    expected_cols = list(qdf.columns) + ['full_res', 'final_ans']

    rdf = pd.read_csv(response_path, encoding='utf-8-sig')

    # if header is broken/missing, reload with no header and assign expected columns
    if 'MCQID' not in rdf.columns or 'final_ans' not in rdf.columns:
        rdf = pd.read_csv(response_path, encoding='utf-8-sig', header=None)
        rdf.columns = expected_cols

    merged = qdf[['MCQID', 'answer_idx']].merge(
        rdf[['MCQID', 'final_ans']],
        on='MCQID',
        how='inner'
    )

    merged['score'] = (
        merged['answer_idx'].astype(str) == merged['final_ans'].astype(str)
    ).astype(int)

    return merged['score'].mean()


def write_summary_row(summary_path, model_name, region, language, final_accuracy):
    row = pd.DataFrame([{
        "model_name": model_name,
        "region": region,
        "language": language,
        "final_accuracy_percentage": round(final_accuracy * 100, 2)
    }])

    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path, encoding="utf-8-sig")

        duplicate_mask = (
            (existing["model_name"] == model_name) &
            (existing["region"] == region) &
            (existing["language"] == language)
        )

        existing = existing[~duplicate_mask]
        updated = pd.concat([existing, row], ignore_index=True)
        updated.to_csv(summary_path, index=False, encoding="utf-8-sig")
    else:
        row.to_csv(summary_path, mode="w", header=True, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--mc_dir', type=str, required=True)
    parser.add_argument('--questions_file', type=str, required=True)
    parser.add_argument('--response_file', type=str, default=None)

    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)

    parser.add_argument('--region', type=str, required=True)
    parser.add_argument('--language', type=str, required=True)

    parser.add_argument('--evaluation_root', type=str, required=True)

    args = parser.parse_args()

    get_model_mc_response(
        model_name=args.model,
        mc_dir=args.mc_dir,
        questions_file=args.questions_file,
        response_file=args.response_file,
        temperature=args.temperature,
        top_p=args.top_p
    )

    final_response_file = args.response_file if args.response_file is not None else f"{args.model}_mc_res.csv"

    score = multiple_choice_score(
        mc_dir=args.mc_dir,
        questions_file=args.questions_file,
        mc_res_file=final_response_file
    )

    print(f"\nFinal accuracy: {score:.4f}")

    region_summary_dir = os.path.join(args.evaluation_root, args.region)
    os.makedirs(region_summary_dir, exist_ok=True)
    summary_path = os.path.join(region_summary_dir, "mc_summary_scores.csv")

    write_summary_row(
        summary_path=summary_path,
        model_name=args.model,
        region=args.region,
        language=args.language,
        final_accuracy=score
    )