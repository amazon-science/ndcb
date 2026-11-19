#!/usr/bin/env bash
# Run GPT-OSS 120B, Mistral Large 3, Qwen3-32B, and GPT-6 Astra on Bedrock.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/evaluate_bedrock_models.sh [--dry-run]

Models: GPT-OSS 120B, Mistral Large 3, Qwen3-32B, then GPT-6 Astra.

For each model: generate with 2,000 tokens, judge with Claude Fable 5.2,
then write a scorecard. Uses all domains, both framings, and three repetitions.
The judge uses the harness's 400-token limit and API-default sampling.

Run inside your activated NDCB environment with working AWS credentials.
AWS_REGION / AWS_DEFAULT_REGION / AWS_PROFILE are handled by NDCB normally.

Optional environment variables:
  NDCB_WORKERS       Concurrent calls within each stage (default: 4).
  NDCB_JUDGE_REGION  Use a separate region for the Fable judge.
  NDCB_RUN_SUFFIX    Append a tag to output names for a fresh experiment.

--dry-run prints commands without running NDCB or creating files.
Rerun with the same settings to resume; successful records are reused.
Individual failed calls remain recorded and do not prevent scoring.
A command-level error stops the batch so you can fix it and resume.
EOF
}

dry_run=0
case "${1:-}" in
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) usage >&2; exit 2 ;;
esac
if (( $# > 0 )); then
    usage >&2
    exit 2
fi

workers="${NDCB_WORKERS:-4}"
suffix="${NDCB_RUN_SUFFIX:-}"
if [[ ! "$workers" =~ ^[1-9][0-9]*$ ]]; then
    printf 'NDCB_WORKERS must be a positive integer.\n' >&2
    exit 2
fi
if [[ -n "$suffix" && ! "$suffix" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    printf 'NDCB_RUN_SUFFIX must contain only letters, digits, dots, underscores, or hyphens.\n' >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd -- "$repo_root"

tag="bedrock-t2000-judge-fable52${suffix:+-$suffix}"
judge_args=(--api bedrock --judge-model us.anthropic.claude-fable-5-2)
if [[ -n "${NDCB_JUDGE_REGION:-}" ]]; then
    judge_args+=(--region "$NDCB_JUDGE_REGION")
fi

if (( ! dry_run )) && ! command -v ndcb >/dev/null 2>&1; then
    printf "ndcb is unavailable. Activate your NDCB environment and install with: pip install -e '.[bedrock]'\n" >&2
    exit 1
fi

trap 'status=$?; printf "\nBatch stopped (exit %s). Fix the command error, then rerun with the same settings to resume.\n" "$status" >&2; exit "$status"' ERR

run_step() {
    local log_file="$1"
    shift
    local command_line
    printf -v command_line ' %q' "$@"
    printf '\n$%s\n' "$command_line"
    if (( dry_run )); then
        return 0
    fi
    mkdir -p -- "$(dirname -- "$log_file")"
    printf '\n$%s\n' "$command_line" >> "$log_file"
    "$@" 2>&1 | tee -a -- "$log_file"
}

run_step "runs/batch-${tag}/preflight.log" ndcb download --check

models=(
    'gpt-oss-120b|GPT-OSS 120B|openai.gpt-oss-120b-1:0'
    'mistral-large-3|Mistral Large 3|mistral.mistral-large-3-675b-instruct'
    'qwen3-32b|Qwen3-32B|qwen.qwen3-32b-v1:0'
    'gpt6-astra|GPT-6 Astra|us.openai.gpt-6-astra'
)
scorecards=()

for model in "${models[@]}"; do
    IFS='|' read -r slug label model_id <<< "$model"
    out="runs/${slug}-${tag}"
    scorecard="results/submissions/${slug}-${tag}.json"

    printf '\nStarting %s\n' "$label"
    run_step "$out/generation.log" ndcb run \
        --api bedrock --model "$model_id" --out "$out" \
        --max-tokens 2000 --temperature 0 --runs 3 --workers "$workers"

    run_step "$out/judge.log" ndcb judge \
        "${judge_args[@]}" --out "$out" --workers "$workers"

    run_step "$out/score.log" ndcb score \
        --out "$out" --model-id "$model_id" \
        --model "$label (Bedrock, max_tokens=2000, judge=Fable 5.2 on Bedrock)" \
        --ci-method bootstrap --bootstrap-iters 10000 --seed 42 \
        --json "$scorecard"
    scorecards+=("$scorecard")
done

if (( dry_run )); then
    printf '\nDry run complete. Scorecards would be written to:\n'
else
    printf '\nBatch finished. Scorecards:\n'
fi
printf '  %s\n' "${scorecards[@]}"
printf '\nScores retain missing-call counts; check each score.log for PARTIAL RUN.\n'
printf 'These settings differ from the paper: 2,000 generation tokens and a Fable 5.2 judge.\n'
