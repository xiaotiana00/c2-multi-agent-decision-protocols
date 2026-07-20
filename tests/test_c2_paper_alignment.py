import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import main as experiment_main
from c2_decision_utils import (
    METRIC_FIELDS,
    OPTIONS,
    _debate_mode_adjustments,
    aggregate_decision,
    aggregate_vote,
    build_option_profiles,
    parse_c2_response,
)
from data.c2_decision import load_data as load_c2_data
from model import model_utils
from scripts import run_c2_paper
from scripts.summarize_c2_runs import history_metrics


def _args(**overrides):
    values = {
        "data": "c2_decision",
        "multi_persona": False,
        "num_agents": 3,
        "c2_temperature": 0.14,
        "c2_top_p": 0.78,
        "c2_top_k": 8,
        "c2_max_new_tokens": 64,
        "seed": 42,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _sample():
    return {
        "id": "c2v4_test",
        "shared_scenario": "Shared scenario text.",
        "commander_note": "COMMANDER_PRIVATE",
        "intel_note": "INTEL_PRIVATE",
        "safety_note": "SAFETY_PRIVATE",
        "options": {"A": "confirm", "B": "wait", "C": "intercept", "D": "strike"},
        "label": "A",
    }


def _response(choice, confidence=0.8, reason="short reason", veto="NONE"):
    return (
        f"{{final answer: ({choice})}}\n"
        f"Reason: {reason}\n"
        f"Confidence: {confidence}\n"
        f"SafetyVeto: {veto}"
    )


def _method_value(tokens, flag):
    index = tokens.index(flag)
    return tokens[index + 1]


def test_paper_protocol_controls_and_generation_settings_are_exact():
    expected = {
        "single_vote": (1, 0, "vote", False),
        "multi_vote": (3, 0, "vote", False),
        "multi_debate_vote_t1": (3, 1, "vote", False),
        "multi_debate_vote_t2": (3, 2, "vote", False),
        "multi_debate_vote_t3": (3, 3, "vote", False),
        "multi_decision": (3, 0, "decision", True),
        "multi_decision_debate_t1": (3, 1, "decision", True),
        "multi_decision_debate_t2": (3, 2, "decision", True),
        "multi_decision_debate_t3": (3, 3, "decision", True),
    }
    assert set(run_c2_paper.MAIN_METHODS) == set(expected)
    assert set(run_c2_paper.METHODS) == set(expected) | {"mdd_t3_no_veto", "mdd_t3_no_role"}
    assert run_c2_paper.QWEN_METHODS == run_c2_paper.MAIN_METHODS + run_c2_paper.ABLATION_METHODS
    assert run_c2_paper.DEEPSEEK_METHODS == (
        "single_vote",
        "multi_vote",
        "multi_debate_vote_t3",
        "multi_decision",
        "multi_decision_debate_t3",
        "mdd_t3_no_veto",
        "mdd_t3_no_role",
    )
    for method, controls in expected.items():
        tokens = run_c2_paper.METHODS[method]
        actual = (
            int(_method_value(tokens, "--num_agents")),
            int(_method_value(tokens, "--debate_rounds")),
            _method_value(tokens, "--solver"),
            "--multi_persona" in tokens,
        )
        assert actual == controls
        assert float(_method_value(tokens, "--c2_temperature")) == 0.14
        assert float(_method_value(tokens, "--c2_top_p")) == 0.78
        assert int(_method_value(tokens, "--c2_top_k")) == 8
        assert int(_method_value(tokens, "--c2_max_new_tokens")) == 64
        assert "--c2_profile_source" not in tokens


def test_homogeneous_prompt_is_replicated_for_three_agents():
    messages = experiment_main._build_initial_messages(
        _args(), _sample(), {"None": ""}, ["None"], ""
    )
    assert len(messages) == 2
    assert all(isinstance(message, dict) for message in messages)

    class FakeChat:
        def __init__(self):
            self.calls = []

        def chat(self, call_messages, args, idx=0):
            self.calls.append((call_messages, idx))
            return _response("A", 0.70 + idx / 100)

    agent = FakeChat()
    outputs = model_utils.engine(messages, agent, _args(), num_agents=3)
    assert len(outputs) == 3
    assert [index for _, index in agent.calls] == [0, 1, 2]
    assert all(call_messages == messages for call_messages, _ in agent.calls)


def test_debate_shares_only_peer_choice_and_reason_and_keeps_notes_private():
    args = _args(multi_persona=True)
    responses = {
        "run__Commander__Agent1": _response("A", 0.81, "commander reason", "D"),
        "run__IntelligenceOfficer__Agent2": _response("B", 0.91, "intel reason", "C"),
        "run__SafetyOfficer__Agent3": _response("C", 0.73, "safety reason", "B"),
    }
    personas = {
        "Commander": "COMMANDER_SYSTEM",
        "IntelligenceOfficer": "INTEL_SYSTEM",
        "SafetyOfficer": "SAFETY_SYSTEM",
    }
    messages = experiment_main.get_new_message(args, _sample(), responses, personas, suffix="")
    commander_text = messages["run__Commander__Agent1"][1]["content"]
    assert "COMMANDER_PRIVATE" in commander_text
    assert "INTEL_PRIVATE" not in commander_text
    assert "SAFETY_PRIVATE" not in commander_text
    assert "intel reason" in commander_text and "safety reason" in commander_text
    assert "Confidence=0.91" not in commander_text
    assert "SafetyVeto=C" not in commander_text
    assert "Confidence=0.81" in commander_text
    assert "SafetyVeto=D" in commander_text


def test_parser_defaults_missing_confidence_and_rejects_bare_letters():
    parsed = parse_c2_response("{final answer: (B)}\nReason: concise")
    assert parsed["choice"] == "(B)"
    assert parsed["confidence"] == 0.68
    assert parsed["recoverable"] is True
    assert parsed["schema_valid"] is False
    assert parsed["confidence_present"] is False
    assert parsed["veto_field_present"] is False
    assert parse_c2_response("A")["choice"] == ""
    assert parse_c2_response("unstructured response")["recoverable"] is False


def test_confidence_weighted_vote_and_consensus_bonus_match_paper_rule():
    responses = {
        "agent1": _response("A", 0.8),
        "agent2": _response("A", 0.6),
        "agent3": _response("B", 1.0),
    }
    _, answer, diagnostics = aggregate_vote(responses)
    assert answer == "(A)"
    assert diagnostics["counts"] == {"(A)": 2, "(B)": 1}
    assert diagnostics["consensus_bonus"]["(A)"] == pytest.approx(0.035)
    assert diagnostics["weighted_counts"]["(A)"] == pytest.approx(1.435)
    assert diagnostics["support_roles"]["(A)"] == ["None"]


def test_answer_change_uses_only_agents_valid_in_both_rounds():
    assert experiment_main._answer_change_rate(["(A)", "", "(C)"], ["(B)", "(A)", ""]) == 1.0
    assert experiment_main._answer_change_rate(["", ""], ["(A)", "(B)"]) == 0.0


def test_heuristic_profiles_ignore_gold_numeric_profiles_and_hidden_weights():
    sample = _sample()
    for metric in METRIC_FIELDS:
        sample[metric] = {option: (0.99 if option == "A" else 0.01) for option in OPTIONS}
    sample["utility_weights"] = {metric: 999.0 for metric in METRIC_FIELDS}

    profiles, metadata = build_option_profiles(sample, return_metadata=True)
    assert metadata["profile_source"] == "heuristic"
    assert profiles["mission_gain"]["A"] == pytest.approx(0.52)

    responses = {
        "run__Commander__Agent1": _response("A", 0.8),
        "run__IntelligenceOfficer__Agent2": _response("B", 0.8),
        "run__SafetyOfficer__Agent3": _response("C", 0.8),
    }
    sample_without_hidden_weights = dict(sample)
    sample_without_hidden_weights.pop("utility_weights")
    _, answer_a, diag_a = aggregate_decision(responses, sample)
    _, answer_b, diag_b = aggregate_decision(
        responses, sample_without_hidden_weights
    )
    assert answer_a == answer_b
    assert diag_a["aggregated_score"] == diag_b["aggregated_score"]


def test_each_heuristic_cue_group_is_applied_once_even_with_repeated_keywords():
    base = _sample()
    repeated = dict(base)
    repeated["options"] = dict(base["options"], A="确认 确认 跟踪 交叉验证")
    unique = dict(base)
    unique["options"] = dict(base["options"], A="确认 跟踪 交叉验证")
    repeated_profiles = build_option_profiles(repeated)
    unique_profiles = build_option_profiles(unique)
    for metric in METRIC_FIELDS:
        assert repeated_profiles[metric]["A"] == unique_profiles[metric]["A"]


def test_no_veto_ablation_keeps_catastrophic_risk_filter():
    sample = _sample()
    sample["options"] = {
        "A": "确认并跟踪，采取有限、可逆的行动",
        "B": "立即直接摧毁目标",
        "C": "实施无差别火力覆盖",
        "D": "使用全部导弹进行饱和攻击",
    }
    responses = {
        "run__Commander__Agent1": _response("B", 0.9),
        "run__IntelligenceOfficer__Agent2": _response("B", 0.9),
        "run__SafetyOfficer__Agent3": _response("B", 0.9, veto="A"),
    }
    _, answer, diagnostics = aggregate_decision(
        responses,
        sample,
        epsilon=0.25,
        safety_veto=False,
    )
    assert diagnostics["safety_vetoes"] == []
    assert diagnostics["feasible"] == ["A"]
    assert answer == "(A)"


def test_debate_adjustments_respect_declared_caps():
    sample = _sample()
    sample["shared_scenario"] = "不明目标接近港区，仅剩10分钟。"
    option_texts = {"A": "确认并跟踪后进行有限拦截", "B": "等待", "C": "火力压制", "D": "中立船作为诱饵"}
    score = {option: 0.0 for option in OPTIONS}
    avg_risk = {option: 0.2 for option in OPTIONS}
    support_count = {"A": 3, "B": 1, "C": 1, "D": 1}
    support_roles = {
        "A": {"Commander", "IntelligenceOfficer", "SafetyOfficer"},
        "B": {"IntelligenceOfficer"},
        "C": {"Commander"},
        "D": {"Commander"},
    }
    support_conf_sum = {"A": 2.4, "B": 0.8, "C": 0.8, "D": 0.8}
    bonuses, penalties = _debate_mode_adjustments(
        sample,
        option_texts,
        score,
        avg_risk,
        support_count,
        support_roles,
        support_conf_sum,
    )
    assert bonuses["A"] <= 0.16 * 0.8 + 1e-9
    assert penalties["D"] <= 0.15 * 0.8 + 1e-9


def test_native_ollama_request_contains_exact_paper_generation_options(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"message": {"content": _response("A")}}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(model_utils.urllib.request, "urlopen", fake_urlopen)
    args = _args(
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="deepseek-r1:7b",
    )
    wrapper = model_utils.OllamaChatWrapper(args)
    output = wrapper.chat([{"role": "user", "content": "C2 Reason:"}], args, idx=2)
    assert "final answer" in output
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["payload"]["model"] == "deepseek-r1:7b"
    assert captured["payload"]["options"]["num_predict"] == 64
    assert captured["payload"]["options"]["temperature"] == 0.14
    assert captured["payload"]["options"]["top_p"] == 0.78
    assert captured["payload"]["options"]["top_k"] == 8
    assert isinstance(captured["payload"]["options"]["seed"], int)


def test_v4_dataset_identity_count_types_and_checksum_are_verified():
    args = SimpleNamespace(
        c2_data_file=None,
        c2_version="v4",
        data_dir=None,
        data_size=0,
        seed=42,
    )
    rows, labels = load_c2_data(args)
    assert len(rows) == len(labels) == 100
    assert len({row["id"] for row in rows}) == 100
    assert len({row["task_type"] for row in rows}) == 20
    dataset_path = Path(args._c2_loaded_file)
    assert dataset_path.name == "c2_decision_v4.jsonl"
    assert args._c2_dataset_sha256 == hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    metadata = json.loads(Path(args._c2_dataset_metadata_file).read_text(encoding="utf-8"))
    assert metadata["dataset_sha256"] == args._c2_dataset_sha256
    assert metadata["preliminary_drafting_checkpoint"] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def test_old_histories_without_parser_audit_fields_are_not_counted_as_failures(tmp_path):
    rounds = {
        "0": {
            "aggregate_answer_iscorr": True,
            "risk_hit": False,
            "catastrophic_hit": False,
            "disagreement_rate": 0.0,
            "answer_change_rate": 0.0,
            "parsed_responses": {"agent": {"choice": "(A)"}},
        }
    }
    history = tmp_path / "old_history.jsonl"
    history.write_text(json.dumps({"sample_id": "old", "rounds": rounds}) + "\n", encoding="utf-8")
    metrics = history_metrics(str(history))
    assert metrics["total_final_agent_outputs"] == 0
    assert metrics["unrecoverable_final_outputs"] == 0
    assert metrics["schema_invalid_final_outputs"] == 0
