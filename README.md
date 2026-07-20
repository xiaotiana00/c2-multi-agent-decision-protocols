# C2 Multi-Agent Decision Protocols

This repository contains the implementation used for the controlled C2 protocol comparison.

## Repository Contents

- `src/`: C2 dataset loading, prompts, aggregation, adjudication, and model backends.
- `src/data/c2/`: the 100-scenario C2 v4 dataset and its integrity metadata.
- `scripts/`: experiment runners, manifest checks, dataset checks, and result summarization.
- `tests/`: protocol-alignment and implementation tests.

Generated histories, model caches, local credentials, and experiment output directories are excluded from version control.

## Models

- `Qwen/Qwen2.5-7B-Instruct`, loaded with NF4 4-bit quantization, double quantization, and FP16 computation.
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, served locally by Ollama as `deepseek-r1:7b` with Q4_K_M quantization.

Both model runs use temperature 0.14, top-p 0.78, top-k 8, a 64-token output limit, and seeds 42, 43, and 44.

## Protocols

The implemented protocols are SV, MV, MDV-T1/T2/T3, MD, and MDD-T1/T2/T3. The component analysis contains MDD-T3 without veto exclusion and MDD-T3 without role conditioning.

Role-conditioned agents receive the shared scenario and only their own private note. During debate, peers exchange their previous choices and short reasons. The final role-based adjudicator is deterministic, uses the fixed risk threshold 0.25, and does not invoke another model.

## Data Boundary

The reported runs use `src/data/c2/c2_decision_v4.jsonl`. Reference actions, high-risk labels, catastrophic-risk labels, and human-assigned numeric profiles are used for construction and evaluation only. Main decision runs derive heuristic profiles from scenario text, candidate actions, and structured agent responses.

## Running the Experiments

Install the Python dependencies, then run the Qwen protocols:

```powershell
python scripts/run_c2_paper.py --model qwen2.5-7b --out_dir out_paper_qwen
python scripts/summarize_c2_runs.py --manifest out_paper_qwen/paper_manifest.jsonl --out_dir out_paper_qwen
```

For the locally installed DeepSeek checkpoint:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_deepseek_ollama_c2.ps1
```

The Qwen command runs the nine main rows and both reported ablations. The Ollama command runs the five reported core rows and both ablations. The Ollama runner records the installed model digest, quantization, architecture, context length, embedding length, and runtime version. It warns if the runtime differs from the Ollama 0.24.0 environment recorded for the historical experiments.

## Verification

```powershell
python -m pytest -q
python scripts/audit_c2_dataset.py --data_file src/data/c2/c2_decision_v4.jsonl --epsilon 0.25
python scripts/audit_c2_manifest.py out_paper_qwen/paper_manifest.jsonl
```
