param(
    [string]$OutDir = "out_paper_deepseek_ollama",
    [string]$Methods = "single_vote,multi_vote,multi_debate_vote_t3,multi_decision,multi_decision_debate_t3,mdd_t3_no_veto,mdd_t3_no_role",
    [string]$Seeds = "42,43,44",
    [int]$DataSize = 100
)

$ErrorActionPreference = "Stop"

$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
$env:OLLAMA_MODEL = "deepseek-r1:7b"
$reportedOllamaVersion = "0.24.0"

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama is not installed or is not available on PATH."
}
$ollamaDetails = ollama show $env:OLLAMA_MODEL
$env:OLLAMA_VERSION = ((ollama --version) -replace '^ollama version is\s+', '').Trim()
$ollamaListLine = ollama list | Where-Object { $_ -match '^deepseek-r1:7b\s+' } | Select-Object -First 1
if (-not $ollamaListLine) {
    throw "Unable to determine the installed digest for $env:OLLAMA_MODEL."
}
$env:OLLAMA_MODEL_DIGEST = ($ollamaListLine -split '\s+')[1]
$quantizationLine = $ollamaDetails | Where-Object { $_ -match '^\s*quantization\s+([^\s]+)' } | Select-Object -First 1
if (-not $quantizationLine) {
    throw "Unable to determine the quantization for $env:OLLAMA_MODEL."
}
$env:OLLAMA_QUANTIZATION = [regex]::Match($quantizationLine, '^\s*quantization\s+([^\s]+)').Groups[1].Value
$env:OLLAMA_ARCHITECTURE = [regex]::Match(($ollamaDetails | Where-Object { $_ -match '^\s*architecture\s+' } | Select-Object -First 1), '^\s*architecture\s+([^\s]+)').Groups[1].Value
$env:OLLAMA_PARAMETER_COUNT = [regex]::Match(($ollamaDetails | Where-Object { $_ -match '^\s*parameters\s+' } | Select-Object -First 1), '^\s*parameters\s+([^\s]+)').Groups[1].Value
$env:OLLAMA_CONTEXT_LENGTH = [regex]::Match(($ollamaDetails | Where-Object { $_ -match '^\s*context length\s+' } | Select-Object -First 1), '^\s*context length\s+([^\s]+)').Groups[1].Value
$env:OLLAMA_EMBEDDING_LENGTH = [regex]::Match(($ollamaDetails | Where-Object { $_ -match '^\s*embedding length\s+' } | Select-Object -First 1), '^\s*embedding length\s+([^\s]+)').Groups[1].Value

if ($env:OLLAMA_VERSION -ne $reportedOllamaVersion) {
    Write-Warning "The manuscript reports Ollama $reportedOllamaVersion, but this host has $env:OLLAMA_VERSION. The actual version will be retained in the internal run record."
}

python scripts/run_c2_paper.py `
    --model ollama `
    --out_dir $OutDir `
    --methods $Methods `
    --seed_list $Seeds `
    --data_size $DataSize `
    --extra_args "--ollama_base_url $env:OLLAMA_BASE_URL --ollama_model $env:OLLAMA_MODEL --ollama_quantization $env:OLLAMA_QUANTIZATION --ollama_model_digest $env:OLLAMA_MODEL_DIGEST --ollama_version $env:OLLAMA_VERSION --ollama_architecture $env:OLLAMA_ARCHITECTURE --ollama_parameter_count $env:OLLAMA_PARAMETER_COUNT --ollama_context_length $env:OLLAMA_CONTEXT_LENGTH --ollama_embedding_length $env:OLLAMA_EMBEDDING_LENGTH --formal_checkpoint deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

python scripts/audit_c2_manifest.py "$OutDir/paper_manifest.jsonl"
python scripts/summarize_c2_runs.py --manifest "$OutDir/paper_manifest.jsonl" --out_dir $OutDir
