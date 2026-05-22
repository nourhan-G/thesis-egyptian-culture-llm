import os
import json
import argparse
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm

from evaluation_utils import *



ANSWER_LIST_KEY = "annotations"
IDK_KEY = "idks"
COUNT_KEY = "count"


def no_common_word(s1, s2):
    if is_float(s1) and is_float(s2):
        return float(s1) != float(s2)

    s1 = str(s1)
    s2 = str(s2)

    if s1 in s2 or s2 in s1:
        return False

    def split_words(a):
        if '/' in a:
            a = a.split('/')
        else:
            a = [a]

        words = []
        for item in a:
            words += item.split()
        return words

    s1_words = split_words(s1.lower())
    s2_words = split_words(s2.lower())

    return len(set(s1_words) & set(s2_words)) == 0


def another_similar_term(question, answers, word, region1, region2):
    """
    Returns True if the candidate distractor is too similar to the correct answer(s)
    and should be blocked.
    """
    if isinstance(answers, str):
        answers = [answers]

    simple_flag = False
    all_float_time_date = True

    for c in answers:
        if is_float(c) and is_float(word):
            if float(c) == float(word):
                simple_flag = True
                break
        elif (is_date_format(c) and is_date_format(word)) or (is_time_format(c) and is_time_format(word)):
            if str(c) in str(word) or str(word) in str(c):
                simple_flag = True
                break
        else:
            all_float_time_date = False

    if simple_flag:
        return True

    if all_float_time_date:
        return False

    prompt = """Determine if a 'target' word is the same in meaning (for example: football and soccer),
or if one is a subset/superset of the other (for example: fruit and apple).

If they are similar in that way, return:
{{"result":"O"}}

If they are different and can both appear as separate MCQ choices, return:
{{"result":"X"}}

Question: {question}
Correct answer(s): {answers}
Correct-answer region: {region1}
Candidate distractor: {word}
Candidate region: {region2}

Write reasoning first, then only one final JSON object.
""".format(
        question=question,
        answers=json.dumps(answers, ensure_ascii=False),
        region1=region1,
        word=word,
        region2=region2
    )

    res = run_openai(prompt, "gpt-5.4", temperature=0, top_p=1.0)
    res = res.replace('{result:', '{"result":')

    json_res = get_json_str(res)

    if isinstance(json_res, dict) and 'result' in json_res:
        return json_res['result'] == 'O'

    return True


def filter_mc_questions(original_questions_df, all_annotations, mc_dir, id_col="ID", answer_text_key="en_answers"):
    """
    Remove a question if:
    1. any region has not-applicable
    2. any region has more than 3 idk-style responses
    3. the top answer in any region has fewer than 2 votes
    """
    filtered_questions_df = original_questions_df.copy()

    for i, row in original_questions_df.iterrows():
        qid = row[id_col]

        has_idk = False
        small_max_vote = False

        for region in all_annotations.keys():
            region_annotation = all_annotations[region]

            if qid not in region_annotation:
                has_idk = True
                print(f"Missing question {qid} in region {region}")
                break

            region_annotation_qid = region_annotation[qid]

            annotations = region_annotation_qid.get(ANSWER_LIST_KEY, [])
            idks = region_annotation_qid.get(IDK_KEY, {})

            if idks.get("not-applicable", 0) > 0 or sum(idks.values()) > 3:
                has_idk = True

            elif (
                annotations
                and len(annotations[0].get(answer_text_key, [])) > 0
                and annotations[0].get(COUNT_KEY, 0) < 2
            ):
                small_max_vote = True

            elif not annotations:
                has_idk = True
                print(f"No annotations for {qid} in region {region}")

            if has_idk or small_max_vote:
                filtered_questions_df = filtered_questions_df.drop(i)
                print(
                    f"Removed question {qid} because of region {region} | "
                    f"has_idk={has_idk} | small_max_vote={small_max_vote}"
                )
                break

    filtered_questions_df.to_csv(
        os.path.join(mc_dir, 'filtered_questions.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    print("Leftover questions:", len(filtered_questions_df))
    return filtered_questions_df


def generate_answer_choices(
    region_list,
    annotation_data_dir,
    annotation_data_template,
    question_dir,
    question_data_template,
    id_col,
    question_col,
    answer_text_key,
    mc_dir,
    output_filename='unique_answer_choice.json',
    dictionary_filename='dictionary.json',
    target_region=None
):
    region_unique_answer_choice = {}

    output_path = os.path.join(mc_dir, output_filename)
    dict_path = os.path.join(mc_dir, dictionary_filename)

    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            region_unique_answer_choice = json.load(f)

    question_source_region = target_region if target_region else region_list[0]

    final_questions = get_questions(
        data_dir=question_dir,
        country=question_source_region,
        template=question_data_template
    )

    all_annotations = {
        region: get_annotations(
            data_dir=annotation_data_dir,
            country=region,
            template=annotation_data_template
        )
        for region in region_list
    }

    filtered_questions = filter_mc_questions(
        final_questions,
        all_annotations,
        mc_dir,
        id_col=id_col,
        answer_text_key=answer_text_key
    )

    same_dict = defaultdict(dict)

    if os.path.exists(dict_path):
        with open(dict_path, 'r', encoding='utf-8') as f:
            old_same_dict = json.load(f)
            for k, v in old_same_dict.items():
                same_dict[k] = v

    target_regions = [target_region] if target_region else region_list

    for _, row in tqdm(filtered_questions.iterrows(), total=len(filtered_questions)):
        qid = row[id_col]

        each_qid_dict = {
            'question': row[question_col],
            'annotations': {}
        }

        print("\nQUESTION:", row[question_col])

        for region in target_regions:
            each_region_dict = {}

            region_qid_data = all_annotations[region].get(qid, {})
            region_annotations = region_qid_data.get(ANSWER_LIST_KEY, [])


            annotations = {
                data[answer_text_key][0]: data[COUNT_KEY]
                for data in region_annotations
                if answer_text_key in data and len(data[answer_text_key]) > 0
            }

            blocked = set()

            if annotations:
                max_vote = max(annotations.values())
                each_region_dict['answer'] = [k for k, v in annotations.items() if v == max_vote]

                choices = {}

                other_regions_annotations = {
                    other_region: {
                        data[answer_text_key][0]: data[COUNT_KEY]
                        for data in all_annotations[other_region].get(qid, {}).get(ANSWER_LIST_KEY, [])
                        if answer_text_key in data and len(data[answer_text_key]) > 0
                    }
                    for other_region in region_list if other_region != region
                }

                all_answer_choices = sorted(
                    [
                        (vote_count, answer, other_region)
                        for other_region, other_annotations in other_regions_annotations.items()
                        for answer, vote_count in other_annotations.items()
                        if vote_count >= 2
                    ],
                    key=lambda x: x[0],
                    reverse=True
                )

                for vote_count, answer, other_region in all_answer_choices:
                    if other_region in choices:
                        continue

                    if answer in blocked:
                        continue

                    flag = True
                    for candidate in annotations.keys():
                        if candidate in same_dict and answer in same_dict[candidate] and same_dict[candidate][answer]:
                            flag = False
                            break

                        flag = no_common_word(answer, candidate)

                        if not flag:
                            same_dict[candidate][answer] = True
                            same_dict[answer][candidate] = True
                            blocked.add(answer)
                            break

                    if flag:
                        final_flag = True
                        for chosen_region, chosen_answer in choices.items():
                            if chosen_answer in same_dict and answer in same_dict[chosen_answer]:
                                if same_dict[chosen_answer][answer]:
                                    blocked.add(answer)
                                    final_flag = False
                                    break
                                else:
                                    continue

                            if (not no_common_word(chosen_answer, answer)) or another_similar_term(
                                each_qid_dict['question'],
                                chosen_answer,
                                answer,
                                chosen_region,
                                other_region
                            ):
                                final_flag = False
                                same_dict[answer][chosen_answer] = True
                                same_dict[chosen_answer][answer] = True
                                blocked.add(answer)
                            else:
                                same_dict[answer][chosen_answer] = False
                                same_dict[chosen_answer][answer] = False

                            if not final_flag:
                                break

                        if final_flag:
                            all_checked = True
                            at_least_one = False

                            for candidate in annotations.keys():
                                if not (candidate in same_dict and answer in same_dict[candidate]):
                                    all_checked = False

                                if candidate in same_dict and answer in same_dict[candidate] and same_dict[candidate][answer]:
                                    at_least_one = True

                                if not all_checked or at_least_one:
                                    break

                            if at_least_one:
                                blocked.add(answer)
                                continue
                            elif all_checked or not another_similar_term(
                                each_qid_dict['question'],
                                list(annotations.keys()),
                                answer,
                                region,
                                other_region
                            ):
                                choices[other_region] = answer

                                if not all_checked:
                                    for candidate in annotations.keys():
                                        same_dict[candidate][answer] = False
                                        same_dict[answer][candidate] = False
                            else:
                                blocked.add(answer)

                each_region_dict['choices'] = choices
                each_qid_dict[region] = each_region_dict

                with open(dict_path, 'w', encoding='utf-8') as f:
                    json.dump(same_dict, f, indent=4, ensure_ascii=False)

            each_qid_dict['annotations'][region] = annotations

        region_unique_answer_choice[qid] = each_qid_dict

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(region_unique_answer_choice, f, indent=4, ensure_ascii=False)


def get_dummy_choices(question, annotations, num):
    prompt = (
        f'Provide {num} dummy option(s) that make sense as possible answers to the question, '
        f'but are factually wrong, exist in real life, and are NOT any real answer from the regions. '
        f'Return JSON only as: {{"dummy_options":[]}}\n\n'
    )

    all_real_answers = list(set([v for region in annotations for v in annotations[region]]))
    json_str = json.dumps(
        {
            'question': question,
            'answers': all_real_answers
        },
        ensure_ascii=False,
        indent=4
    )
    prompt += json_str

    while True:
        res = run_openai(prompt, "gpt-5.4", temperature=1, top_p=1.0)
        res = res.replace('{dummy_options:', '{"dummy_options":')
        json_res = get_json_str(res)

        if (
            isinstance(json_res, dict)
            and 'dummy_options' in json_res
            and isinstance(json_res['dummy_options'], list)
            and len(json_res['dummy_options']) == num
            and len(set(json_res['dummy_options'])) == num
            and len(set(json_res['dummy_options']) & set(all_real_answers)) == 0
        ):
            return [s.lower() for s in json_res['dummy_options']]


def generate_prompt_mc(question, region, answers, choices, min_choice, dummy_choices, lang):
    res = []

    for answer in answers:
        if lang == "ar":
            prompt = (
                f'{question} من غير أي شرح، اختار حرف واحد بس من الاختيارات اللي جاية '
                f'(زي A أو B أو C). اكتب الإجابة بصيغة JSON بالشكل ده: {{"answer_choice":""}}\n\n'
            )
            answer_label = '\nالإجابة:'

        else:
            prompt = (
                f'{question} Without any explanation, choose only one from the given alphabet choices '
                f'(e.g., A, B, C). Provide as JSON format: {{"answer_choice":""}}\n\n'
            )
            answer_label = '\nAnswer:'

        for chosen_choices in combinations(choices.items(), min_choice):
            all_choices = sorted(
                [(v, k) for k, v in chosen_choices] +
                [(answer, region)] +
                [(dummy, 'dummy') for dummy in dummy_choices],
                key=lambda x: x[0].lower()
            )

            all_choices_idx = {}
            all_choices_region = {}
            answer_idx = -1

            this_prompt = prompt
            for i, (choice_text, choice_region) in enumerate(all_choices):
                letter = chr(ord('A') + i)

                if choice_text == answer:
                    answer_idx = letter

                all_choices_idx[letter] = choice_text
                all_choices_region[letter] = choice_region
                this_prompt += f'{letter}. {choice_text}\n'

            this_prompt += answer_label
            res.append((this_prompt, all_choices_idx, all_choices_region, answer_idx))

    return res

def generate_multiple_choice(
    region_list,
    mc_dir,
    answer_choice_file,
    questions_file,
    lang,
    entity_col_name="country",
    generate_dummy=True,
    target_region=None
):
    with open(os.path.join(mc_dir, answer_choice_file), 'r', encoding='utf-8') as f:
        answer_choices = json.load(f)

    questions_path = os.path.join(mc_dir, questions_file)
    if os.path.exists(questions_path):
        os.remove(questions_path)

    write_csv_row(
        ['MCQID', 'ID', entity_col_name, 'prompt', 'choices', 'choice_countries', 'answer_idx'],
        questions_path
    )

    pb = tqdm(answer_choices, total=len(answer_choices))
    for qid in pb:
        pb.set_description(qid)
        question = answer_choices[qid]['question']
        cnt = 0

        valid_regions = [
            region for region in region_list
            if region in answer_choices[qid] and 'choices' in answer_choices[qid][region]
        ]

        if target_region is not None:
            valid_regions = [region for region in valid_regions if region == target_region]

        if not valid_regions:
            continue

        min_choice = min([len(answer_choices[qid][region]['choices']) for region in valid_regions])

        dummy_choices = []

        for region in valid_regions:
            if min_choice < 3:
                if generate_dummy and 'dummy_choices' not in answer_choices[qid][region]:
                    dummy_choices = get_dummy_choices(
                        question,
                        answer_choices[qid]['annotations'],
                        3 - min_choice
                    )
                    answer_choices[qid][region]['dummy_choices'] = dummy_choices
                    with open(os.path.join(mc_dir, answer_choice_file), 'w', encoding='utf-8') as f:
                        json.dump(answer_choices, f, indent=4, ensure_ascii=False)
                elif 'dummy_choices' in answer_choices[qid][region]:
                    dummy_choices = answer_choices[qid][region]['dummy_choices']
                else:
                    print(f'ERROR: No dummy choices for {qid} in {region}')
                    continue

            pb.set_postfix({'region': region})

            prompt_questions = generate_prompt_mc(
                question,
                region,
                answer_choices[qid][region]['answer'],
                answer_choices[qid][region]['choices'],
                min(min_choice, 3),
                dummy_choices,
                lang
            )

            if prompt_questions:
                for q, choices, choice_regions, answer_idx in prompt_questions:
                    write_csv_row(
                        [
                            f'{qid}_{cnt}',
                            qid,
                            region,
                            q,
                            json.dumps(choices, indent=4, ensure_ascii=False),
                            json.dumps(choice_regions, indent=4, ensure_ascii=False),
                            answer_idx
                        ],
                        questions_path
                    )
                    cnt += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_col', type=str, default='ID')
    parser.add_argument('--lang', type=str, choices=['ar', 'en'], required=True)
    parser.add_argument('--target_region', type=str, default=None)
    parser.add_argument('--question_dir', type=str, default='./questions/')
    parser.add_argument('--question_data_template', type=str, default='{country}_questions.csv')
    parser.add_argument('--annotation_dir', type=str, default='./annotations/')
    parser.add_argument('--annotation_data_template', type=str, default='{country}_data.json')
    parser.add_argument('--mc_dir', type=str, default='./mc_data')

    args = parser.parse_args()

    region_list = [
        'GreaterCairo',
        'Alexandria',
        'LowerEgypt',
        'UpperEgypt',
        'Sinai',
        'TheSuezCanalCities'
    ]

    if args.lang == 'ar':
        question_col = 'Question'
        answer_text_key = 'answers'
        lang_suffix = 'ar'
        question_data_template = '{country}_questions.csv'

    else:
        question_col = 'Translation'
        answer_text_key = 'en_answers'
        lang_suffix = 'en'
        question_data_template = '{country}_questions.csv'

    region_suffix = args.target_region if args.target_region else "all"

    if args.target_region:
        run_folder_name = args.target_region
    else:
        run_folder_name = "all_regions"

    run_mc_dir = os.path.join(args.mc_dir, run_folder_name)
    os.makedirs(run_mc_dir, exist_ok=True)

    answer_choice_file = f'tmp_unique_answer_choice_{region_suffix}_{lang_suffix}.json'
    dictionary_file = f'tmp_dictionary_{region_suffix}_{lang_suffix}.json'
    mc_questions_file = f'mc_questions_{region_suffix}_{lang_suffix}.csv'

    generate_answer_choices(
        region_list=region_list,
        annotation_data_dir=args.annotation_dir,
        annotation_data_template=args.annotation_data_template,
        question_dir=args.question_dir,
        question_data_template=question_data_template,
        id_col=args.id_col,
        question_col=question_col,
        answer_text_key=answer_text_key,
        mc_dir=run_mc_dir,
        output_filename=answer_choice_file,
        dictionary_filename=dictionary_file,
        target_region=args.target_region
    )

    generate_multiple_choice(
        region_list=region_list,
        mc_dir=run_mc_dir,
        answer_choice_file=answer_choice_file,
        questions_file=mc_questions_file,
        lang=args.lang,
        entity_col_name='country',
        target_region=args.target_region
    )

    for tmp_file in [answer_choice_file, dictionary_file, 'filtered_questions.csv']:
        tmp_path = os.path.join(run_mc_dir, tmp_file)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

