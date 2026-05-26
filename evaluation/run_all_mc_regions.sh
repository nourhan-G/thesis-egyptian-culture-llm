#!/bin/bash
set -e

MODELS=(
  "gpt-5.4-mini"
  "gpt-5.4"
  "claude-sonnet-4-6"
  "claude-haiku-4-5"
  "claude-opus-4-6"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
  "Qwen/Qwen2.5-72B-Instruct"
  "CohereLabs/aya-expanse-32b"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.3-70B-Instruct"
  "CohereLabs/c4ai-command-r7b-arabic-02-2025"
  "silma-ai/SILMA-9B-Instruct-v1.0"
)

REGIONS=(
  "GreaterCairo"
  "Alexandria"
  "LowerEgypt"
  "UpperEgypt"
  "Sinai"
  "TheSuezCanalCities"
)

LANGUAGES=(
  "en"
  "ar"
  "msa"
)

BASE_DIR="$(pwd)"
BASE_OUTPUT_DIR="${BASE_DIR}/evaluation_result_mc"

mkdir -p "$BASE_OUTPUT_DIR"

for MODEL in "${MODELS[@]}"; do
  SAFE_MODEL="${MODEL//\//_}"

  for REGION in "${REGIONS[@]}"; do
    mkdir -p "${BASE_OUTPUT_DIR}/${REGION}"

    for LANG in "${LANGUAGES[@]}"; do
      if [ "$LANG" = "en" ]; then
        QUESTIONS_FILE="mc_questions_${REGION}_en.csv"
      elif [ "$LANG" = "ar" ]; then
        QUESTIONS_FILE="mc_questions_${REGION}_ar.csv"
      else
        QUESTIONS_FILE="mc_questions_${REGION}_msa.csv"
      fi

      python multiple_choice_evaluation.py \
        --model "$MODEL" \
        --mc_dir "./mc_data/${REGION}" \
        --questions_file "$QUESTIONS_FILE" \
        --response_file "${BASE_OUTPUT_DIR}/${REGION}/${SAFE_MODEL}_mc_res_${REGION}_${LANG}.csv" \
        --region "$REGION" \
        --language "$LANG" \
        --evaluation_root "$BASE_OUTPUT_DIR"
    done
  done
done

echo "MC evaluation done for en, ar, and msa."