import collections
import re
from copy import deepcopy
from typing import Dict, Iterable, Optional

OPTIONS = ("A", "B", "C", "D")
METRIC_FIELDS = (
    "mission_gain",
    "collateral_risk",
    "exposure_risk",
    "delay_cost",
    "roe_risk",
    "catastrophic_risk",
)
RISK_METRICS = tuple(m for m in METRIC_FIELDS if m != "mission_gain")

ROLE_METRIC_WEIGHTS = {
    "Commander": {
        "mission_gain": 1.20,
        "collateral_risk": 0.78,
        "exposure_risk": 0.82,
        "delay_cost": 0.92,
        "roe_risk": 0.70,
        "catastrophic_risk": 1.10,
    },
    "IntelligenceOfficer": {
        "mission_gain": 0.98,
        "collateral_risk": 0.88,
        "exposure_risk": 0.98,
        "delay_cost": 0.72,
        "roe_risk": 0.82,
        "catastrophic_risk": 1.00,
    },
    "SafetyOfficer": {
        "mission_gain": 0.72,
        "collateral_risk": 1.24,
        "exposure_risk": 1.02,
        "delay_cost": 0.50,
        "roe_risk": 1.20,
        "catastrophic_risk": 1.42,
    },
    "None": {
        "mission_gain": 1.00,
        "collateral_risk": 1.00,
        "exposure_risk": 1.00,
        "delay_cost": 1.00,
        "roe_risk": 1.00,
        "catastrophic_risk": 1.00,
    },
}
ROLE_IMPORTANCE = {"Commander": 0.40, "IntelligenceOfficer": 0.30, "SafetyOfficer": 0.30, "None": 1.00}

SEVERE_RISK_PATTERNS = [
    "无差别", "饱和", "全域", "全部", "全口径", "全面", "不考虑", "强制登临", "火力覆盖", "全部导弹",
    "最高战备", "实弹搜索", "持续压制", "直接摧毁", "击落", "命中立", "命民用", "作为干扰手段",
]
CAREFUL_PATTERNS = [
    "确认", "交叉验证", "跟踪", "二次识别", "二次确认", "分区", "分层", "联合", "远距", "疏导",
    "警戒", "可逆", "脱敏", "有限", "迟滞", "定向", "工兵", "无人侦察", "核查", "预警",
]
PASSIVE_PATTERNS = ["完全忽略", "不回应", "放弃", "全部撤回", "等待", "暂停全部", "停止一切", "原地待命"]
PROTECTED_ASSET_PATTERNS = ["平民", "居民", "民用", "中立", "医院", "医疗", "伤员", "渔船", "民船", "友军", "观察员"]
PROTECTED_EXPLOIT_PATTERNS = ["作为干扰", "作为诱饵", "配合靠近观察", "人工拦截", "靠近可疑区域", "目视确认", "徒步先行探路", "自行对空射击应对"]
HIGH_VALUE_PATTERNS = ["补给", "医院", "油料", "雷达站", "高地", "指挥车", "仓储区", "补给线", "炮兵阵地"]
UNCERTAINTY_PATTERNS = ["疑似", "不明", "单次", "不足", "可能", "未确认", "存在诱饵", "需进一步", "短暂", "尚不明确"]
URGENCY_PATTERNS = ["10分钟", "15分钟", "20分钟", "30分钟", "40分钟", "一夜内", "24小时", "仅剩", "短时间内"]

CONGESTED_ZONE_PATTERNS = ["港口运行", "港区", "泊位", "港内", "民船活动", "装卸车辆", "车辆活动", "车流", "人群", "疏散通道"]
CONFIRM_CONTAIN_PATTERNS = ["多源识别", "确认", "交叉验证", "核查", "跟踪", "预警", "局部警戒区", "局部警戒", "划设", "压缩", "分流", "疏导", "伪装", "戒备", "分层", "定向", "有限", "可逆"]
COERCIVE_PREID_PATTERNS = ["压制", "机动拦阻", "迫使目标显形", "先处置", "补做识别", "先打后识别", "直接拦截", "火力", "压制射击"]


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(v)))


def normalize_choice(choice):
    if not choice:
        return ""
    text = str(choice).strip().upper()
    if text in OPTIONS:
        return f"({text})"
    m = re.search(r"([A-D])", text)
    return f"({m.group(1)})" if m else ""


def choice_key(choice):
    c = normalize_choice(choice)
    return c[1] if c else ""


def extract_choice(text: str):
    if not text:
        return ""
    text = str(text).strip()
    patterns = [
        r"\{[^{}]*final\s*(?:answer|choice)\s*[:：]?\s*\(?([A-Da-d])\)?[^{}]*\}?",
        r"(?:final\s*(?:answer|choice)|answer|choice|最终答案|答案|选择)\s*[:：]?\s*\(?([A-Da-d])\)?",
        r"\(([A-Da-d])\)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return normalize_choice(m.group(1))
    return ""


def extract_reason(text: str):
    if not text:
        return ""
    text = str(text).strip()
    patterns = [r"Reason\s*[:：]\s*(.+)", r"理由\s*[:：]\s*(.+)", r"依据\s*[:：]\s*(.+)"]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip().splitlines()[0][:220]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if "final answer" not in ln.lower() and "答案" not in ln:
            return ln[:220]
    return ""


def extract_confidence(text: str, default=0.68):
    if not text:
        return default
    m = re.search(r"Confidence\s*[:：]\s*([-+]?\d*\.?\d+)", str(text), flags=re.IGNORECASE)
    if not m:
        return default
    value = float(m.group(1))
    if value > 1.0:
        value = value / 100.0
    return _clamp(value, 0.05, 1.0)


def extract_vetoes(text: str):
    if not text:
        return set()
    m = re.search(r"SafetyVeto\s*[:：]\s*([A-D,\s]+)", str(text), flags=re.IGNORECASE)
    if not m:
        return set()
    raw = m.group(1)
    return {token.strip().upper() for token in raw.split(",") if token.strip().upper() in OPTIONS}


def parse_c2_response(text: str):
    raw = str(text) if text is not None else ""
    choice = extract_choice(raw)
    reason = extract_reason(raw)
    confidence_present = bool(re.search(r"Confidence\s*[:：]\s*[-+]?\d*\.?\d+", raw, flags=re.IGNORECASE))
    veto_present = bool(re.search(r"SafetyVeto\s*[:：]", raw, flags=re.IGNORECASE))
    return {
        "raw": raw,
        "choice": choice,
        "reason": reason,
        "confidence": extract_confidence(raw),
        "vetoes": extract_vetoes(raw),
        "recoverable": bool(choice),
        "schema_valid": bool(choice and reason and confidence_present and veto_present),
        "confidence_present": confidence_present,
        "veto_field_present": veto_present,
    }


def infer_role(agent_name: str):
    if "__" not in agent_name:
        return "None"
    parts = agent_name.split("__")
    if len(parts) >= 2:
        role = parts[-2]
        if role in ROLE_METRIC_WEIGHTS:
            return role
    return "None"


def _all_context_text(sample: dict):
    sample = sample or {}
    parts = [
        sample.get("shared_scenario", sample.get("scenario", "")),
        sample.get("commander_note", ""),
        sample.get("intel_note", ""),
        sample.get("safety_note", ""),
    ]
    for opt in OPTIONS:
        parts.append(str((sample.get("options") or {}).get(opt, "")))
    return "\n".join([str(p) for p in parts if p])


def _context_flags(sample: dict):
    text = _all_context_text(sample)
    return {
        "protected_present": any(k in text for k in PROTECTED_ASSET_PATTERNS),
        "medical_present": any(k in text for k in ["医疗", "伤员", "医院"]),
        "friendly_present": any(k in text for k in ["友军", "观察员", "巡逻队"]),
        "neutral_present": any(k in text for k in ["中立", "渔船", "民用", "民船"]),
        "uncertain": any(k in text for k in UNCERTAINTY_PATTERNS),
        "urgent": any(k in text for k in URGENCY_PATTERNS),
        "high_value": any(k in text for k in HIGH_VALUE_PATTERNS),
        "congested_zone": any(k in text for k in CONGESTED_ZONE_PATTERNS),
    }


def _base_profiles():
    return {metric: {opt: 0.0 for opt in OPTIONS} for metric in METRIC_FIELDS}


def _option_profiles_from_text(option_text: str, flags: dict):
    text = option_text or ""
    mission = 0.52
    collateral = 0.40
    exposure = 0.40
    delay = 0.36
    roe = 0.36
    catastrophic = 0.36

    if any(kw in text for kw in SEVERE_RISK_PATTERNS):
        collateral += 0.14
        exposure += 0.12
        roe += 0.12
        catastrophic += 0.16

    if any(kw in text for kw in CAREFUL_PATTERNS):
        mission += 0.04
        collateral -= 0.08
        exposure -= 0.08
        catastrophic -= 0.08

    if any(kw in text for kw in PASSIVE_PATTERNS):
        mission -= 0.18
        delay += 0.18
        exposure -= 0.04

    if ("立即" in text or "立刻" in text) and not any(k in text for k in ["确认", "识别", "跟踪", "疏导", "警戒"]):
        collateral += 0.08
        exposure += 0.06
        catastrophic += 0.08

    if any(k in text for k in ["跟踪", "二次识别", "交叉验证", "核查", "疏导", "预警", "分区", "分层", "远距", "工兵", "无人侦察"]):
        mission += 0.06
        collateral -= 0.10
        exposure -= 0.08
        delay -= 0.04
        roe -= 0.06
        catastrophic -= 0.10

    if any(k in text for k in ["有限", "定向", "迟滞", "可逆", "联合"]):
        mission += 0.04
        collateral -= 0.04
        exposure -= 0.04
        catastrophic -= 0.04

    if flags["urgent"] and any(k in text for k in PASSIVE_PATTERNS):
        mission -= 0.08
        delay += 0.10

    if flags["protected_present"] and any(k in text for k in ["无差别", "饱和", "全域", "全口径", "持续压制", "火力覆盖", "全部导弹"]):
        collateral += 0.18
        roe += 0.16
        catastrophic += 0.20

    if any(asset in text for asset in ["中立", "民用", "医疗", "医院", "伤员", "渔船", "民船", "友军", "观察员"]) and any(verb in text for verb in PROTECTED_EXPLOIT_PATTERNS):
        collateral += 0.24
        exposure += 0.16
        roe += 0.24
        catastrophic += 0.42
        mission -= 0.08

    if flags["medical_present"] and any(k in text for k in ["全向防空", "饱和拦截", "全部导弹", "火力覆盖"]):
        collateral += 0.16
        catastrophic += 0.18

    if flags["friendly_present"] and any(k in text for k in ["持续压制", "全口径", "无差别", "靠近可疑区域", "目视确认", "徒步先行"]):
        collateral += 0.12
        catastrophic += 0.16

    if flags["high_value"] and any(k in text for k in ["跟踪", "分层", "迟滞", "有限", "预警", "定向"]):
        mission += 0.04

    return {
        "mission_gain": _clamp(mission),
        "collateral_risk": _clamp(collateral),
        "exposure_risk": _clamp(exposure),
        "delay_cost": _clamp(delay),
        "roe_risk": _clamp(roe),
        "catastrophic_risk": _clamp(catastrophic),
    }



def _option_semantic_flags(option_text: str):
    text = option_text or ""
    return {
        "passive": any(k in text for k in PASSIVE_PATTERNS) and not any(k in text for k in CONFIRM_CONTAIN_PATTERNS),
        "confirm_contain": any(k in text for k in CONFIRM_CONTAIN_PATTERNS),
        "coercive_preid": any(k in text for k in COERCIVE_PREID_PATTERNS),
        "protected_exploit": any(asset in text for asset in ["中立", "民用", "医疗", "医院", "伤员", "渔船", "民船", "友军", "观察员"]) and any(verb in text for verb in PROTECTED_EXPLOIT_PATTERNS),
    }


def _debate_mode_adjustments(
    sample: dict,
    option_texts: Dict[str, str],
    score: Dict[str, float],
    avg_risk: Dict[str, float],
    support_count: Dict[str, int],
    support_roles: Dict[str, set],
    support_conf_sum: Dict[str, float],
):
    flags = _context_flags(sample)
    bonuses = {opt: 0.0 for opt in OPTIONS}
    penalties = {opt: 0.0 for opt in OPTIONS}
    for opt in OPTIONS:
        cnt = int(support_count.get(opt, 0))
        if cnt <= 0:
            continue
        avg_conf = float(support_conf_sum.get(opt, 0.0)) / max(cnt, 1)
        roles = support_roles.get(opt, set()) or set()
        sem = _option_semantic_flags(option_texts.get(opt, ""))
        bonus = 0.0
        penalty = 0.0

        if sem["confirm_contain"]:
            if cnt >= 2 and len(roles) >= 2:
                bonus += 0.16 * avg_conf
            if "SafetyOfficer" in roles:
                bonus += 0.05 * avg_conf
            if "Commander" in roles and "SafetyOfficer" in roles:
                bonus += 0.05 * avg_conf
            if "IntelligenceOfficer" in roles and "SafetyOfficer" in roles:
                bonus += 0.04 * avg_conf
            if flags["uncertain"]:
                bonus += 0.04 * avg_conf
            if flags["congested_zone"] or flags["protected_present"]:
                bonus += 0.04 * avg_conf

        if sem["passive"] and cnt <= 1:
            if flags["urgent"] or flags["high_value"]:
                penalty += 0.08 * avg_conf
            if flags["uncertain"] and ("Commander" not in roles):
                penalty += 0.04 * avg_conf

        if sem["coercive_preid"] and cnt <= 1:
            if flags["uncertain"]:
                penalty += 0.08 * avg_conf
            if flags["congested_zone"] or flags["protected_present"]:
                penalty += 0.05 * avg_conf
            if "SafetyOfficer" not in roles:
                penalty += 0.03 * avg_conf

        if sem["protected_exploit"]:
            penalty += 0.15 * avg_conf

        bonus = min(bonus, 0.16 * avg_conf)
        penalty = min(penalty, 0.15 * avg_conf)
        score[opt] += bonus
        score[opt] -= penalty
        bonuses[opt] = round(bonus, 4)
        penalties[opt] = round(penalty, 4)
    return bonuses, penalties


def _promote_debate_consensus_choice(
    ranked: Iterable[str],
    score: Dict[str, float],
    avg_risk: Dict[str, float],
    support_count: Dict[str, int],
    support_roles: Dict[str, set],
    option_texts: Dict[str, str],
    sample: dict,
):
    ranked = list(ranked)
    if not ranked:
        return ranked
    flags = _context_flags(sample)
    best = ranked[0]
    best_sem = _option_semantic_flags(option_texts.get(best, ""))

    candidates = []
    for opt in ranked:
        cnt = int(support_count.get(opt, 0))
        roles = support_roles.get(opt, set()) or set()
        sem = _option_semantic_flags(option_texts.get(opt, ""))
        if cnt >= 2 and len(roles) >= 2 and "SafetyOfficer" in roles and sem["confirm_contain"]:
            if score[opt] >= score[best] - 0.12 and avg_risk[opt] <= avg_risk[best] + 0.02:
                candidates.append(opt)

    if not candidates and support_count.get(best, 0) <= 1 and best_sem["coercive_preid"] and (flags["uncertain"] or flags["congested_zone"]):
        for opt in ranked:
            cnt = int(support_count.get(opt, 0))
            roles = support_roles.get(opt, set()) or set()
            sem = _option_semantic_flags(option_texts.get(opt, ""))
            if cnt >= 2 and "SafetyOfficer" in roles and sem["confirm_contain"]:
                if score[opt] >= score[best] - 0.10 and avg_risk[opt] <= avg_risk[best] + 0.02:
                    candidates.append(opt)

    if not candidates:
        return ranked

    preferred = sorted(
        set(candidates),
        key=lambda opt: (
            -int("SafetyOfficer" in (support_roles.get(opt, set()) or set())),
            -int("Commander" in (support_roles.get(opt, set()) or set())),
            -support_count.get(opt, 0),
            -score[opt],
            avg_risk[opt],
            opt,
        ),
    )[0]
    return [preferred] + [opt for opt in ranked if opt != preferred]


def build_option_profiles(sample: dict, return_metadata: bool = False):
    sample = sample or {}
    profiles = _base_profiles()
    flags = _context_flags(sample)
    options = sample.get("options") or {}
    for opt in OPTIONS:
        option_text = str(options.get(opt, ""))
        metric_values = _option_profiles_from_text(option_text, flags)
        for metric, value in metric_values.items():
            profiles[metric][opt] = value

    if return_metadata:
        return profiles, {"profile_source": "heuristic"}
    return profiles


def _normalize_metric_weights(metric_weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    base = {metric: 1.0 for metric in METRIC_FIELDS}
    if not metric_weights:
        return base
    for metric, value in metric_weights.items():
        if metric in base:
            base[metric] = float(value)
    return base


def utility_from_profile(role: str, profiles: dict, option: str, metric_weights: Optional[Dict[str, float]] = None):
    role_weights = ROLE_METRIC_WEIGHTS.get(role, ROLE_METRIC_WEIGHTS["None"])
    global_weights = _normalize_metric_weights(metric_weights)
    score = global_weights["mission_gain"] * role_weights["mission_gain"] * profiles["mission_gain"][option]
    for metric in RISK_METRICS:
        score -= global_weights[metric] * role_weights[metric] * profiles[metric][option]
    return score


def aggregate_vote(responses: dict):
    parsed = {name: parse_c2_response(resp) for name, resp in responses.items()}
    final_answers = [parsed_resp["choice"] for parsed_resp in parsed.values()]
    counter = collections.Counter()
    weighted = collections.defaultdict(float)
    per_agent_weights = {}
    support_roles = collections.defaultdict(set)
    support_conf_sum = collections.defaultdict(float)

    for agent_name, parsed_resp in parsed.items():
        choice = normalize_choice(parsed_resp["choice"])
        if not choice:
            continue
        role = "None"
        importance = 1.0
        confidence = float(parsed_resp.get("confidence", 0.68))
        vote_weight = importance * confidence
        counter[choice] += 1
        weighted[choice] += vote_weight
        support_roles[choice].add(role)
        support_conf_sum[choice] += confidence
        per_agent_weights[agent_name] = {
            "role": role,
            "choice": choice,
            "confidence": round(confidence, 4),
            "importance": round(importance, 4),
            "vote_weight": round(vote_weight, 4),
        }

    if not counter:
        return final_answers, "", {"mode": "vote", "counts": {}, "weighted_counts": {}, "consensus_bonus": {}}

    consensus_bonus = {}
    for opt in list(counter.keys()):
        count = counter[opt]
        avg_conf = support_conf_sum[opt] / max(count, 1)
        role_diversity = len(support_roles[opt])
        bonus = 0.0
        if count >= 2:
            bonus += 0.05 * avg_conf
        if role_diversity >= 2:
            bonus += 0.05 * avg_conf
        if count >= max(2, len(parsed)):
            bonus += 0.08 * avg_conf
        weighted[opt] += bonus
        consensus_bonus[opt] = round(bonus, 4)

    ranked = sorted(counter.keys(), key=lambda opt: (-weighted[opt], -counter[opt], opt))
    answer = ranked[0]
    top_weight = weighted[answer]
    tied = [opt for opt in ranked if abs(weighted[opt] - top_weight) <= 1e-9]
    diagnostics = {
        "mode": "vote",
        "counts": dict(counter),
        "weighted_counts": {opt: round(weighted[opt], 4) for opt in counter},
        "consensus_bonus": consensus_bonus,
        "support_count": {opt: int(counter[opt]) for opt in counter},
        "support_roles": {opt: sorted(list(support_roles[opt])) for opt in counter},
        "tied_candidates": tied,
        "per_agent_weights": per_agent_weights,
    }
    return final_answers, answer, diagnostics


def _reason_choice_adjustments(reason: str):
    reason = reason or ""
    utility_delta = 0.0
    catastrophic_delta = 0.0
    collateral_delta = 0.0
    exposure_delta = 0.0
    veto = False

    if any(k in reason for k in ["确认", "交叉验证", "核查", "跟踪", "疏导", "预警", "工兵", "无人侦察", "定向", "分层"]):
        utility_delta += 0.08
        catastrophic_delta -= 0.06
        collateral_delta -= 0.05
        exposure_delta -= 0.04

    if any(k in reason for k in ["无差别", "饱和", "全域", "全部", "全口径", "不考虑", "火力覆盖", "全部导弹"]):
        utility_delta -= 0.18
        catastrophic_delta += 0.16
        collateral_delta += 0.12
        exposure_delta += 0.08

    protected_in_reason = any(k in reason for k in ["医疗", "医院", "伤员", "平民", "居民", "中立", "渔船", "民用", "观察员", "友军"])
    exploit_in_reason = any(k in reason for k in ["作为干扰", "作为诱饵", "配合靠近观察", "人工拦截", "靠近", "目视确认", "徒步"])
    if protected_in_reason and exploit_in_reason:
        utility_delta -= 0.42
        catastrophic_delta += 0.50
        collateral_delta += 0.28
        exposure_delta += 0.18
        veto = True

    return {
        "utility_delta": utility_delta,
        "catastrophic_delta": catastrophic_delta,
        "collateral_delta": collateral_delta,
        "exposure_delta": exposure_delta,
        "veto": veto,
    }


def _auto_option_veto(option_text: str, sample: dict):
    text = option_text or ""
    flags = _context_flags(sample)
    if any(asset in text for asset in ["中立", "民用", "医疗", "医院", "伤员", "渔船", "民船", "友军", "观察员"]) and any(verb in text for verb in PROTECTED_EXPLOIT_PATTERNS):
        return True
    if flags["protected_present"] and any(k in text for k in ["无差别", "饱和", "全域", "全口径", "持续压制", "全部导弹"]):
        return True
    return False


def _rank_feasible_options(feasible: Iterable[str], score: Dict[str, float], avg_risk: Dict[str, float], endorsement: Dict[str, float], vote_key: str, fallback_reason: Optional[str]):
    feasible = list(feasible)
    if not feasible:
        return []

    if fallback_reason == "no_option_satisfies_epsilon":
        min_risk = min(avg_risk[opt] for opt in feasible)
        safe_band = [opt for opt in feasible if avg_risk[opt] <= min_risk + 0.08]
        ranked = sorted(
            safe_band,
            key=lambda opt: (
                -(opt == vote_key),
                avg_risk[opt],
                -score[opt],
                -endorsement[opt],
                opt,
            ),
        )
        return ranked

    return sorted(
        feasible,
        key=lambda opt: (
            -score[opt],
            avg_risk[opt],
            -endorsement[opt],
            -(opt == vote_key),
            opt,
        ),
    )


def aggregate_decision(
    responses: dict,
    sample: dict,
    epsilon=0.25,
    role_importance=None,
    safety_veto=True,
    metric_weights: Optional[Dict[str, float]] = None,
    debate_mode: bool = False,
):
    role_importance = role_importance or ROLE_IMPORTANCE
    parsed = {name: parse_c2_response(resp) for name, resp in responses.items()}
    profiles, profile_meta = build_option_profiles(sample, return_metadata=True)
    option_texts = {opt: str((sample.get("options") or {}).get(opt, "")) for opt in OPTIONS}
    # Gold labels and human-assigned dataset profiles are evaluation-only. The
    # solver accepts only explicit run-level weights, never hidden sample fields.
    metric_weights = _normalize_metric_weights(metric_weights)

    final_answers, vote_answer, vote_diag = aggregate_vote(responses)
    vote_key = choice_key(vote_answer)

    score = {opt: 0.0 for opt in OPTIONS}
    endorsement = {opt: 0.0 for opt in OPTIONS}
    avg_risk = {opt: 0.0 for opt in OPTIONS}
    role_views = {}
    safety_vetoes = set()
    total_importance = 0.0
    support_count = {opt: 0 for opt in OPTIONS}
    support_roles = {opt: set() for opt in OPTIONS}
    support_weight = {opt: 0.0 for opt in OPTIONS}
    support_conf_sum = {opt: 0.0 for opt in OPTIONS}
    safety_supported = {opt: False for opt in OPTIONS}

    if safety_veto:
        for opt in OPTIONS:
            if _auto_option_veto(option_texts[opt], sample):
                safety_vetoes.add(opt)

    for agent_name, parsed_resp in parsed.items():
        role = infer_role(agent_name)
        importance = float(role_importance.get(role, role_importance.get("None", 1.0)))
        confidence = parsed_resp["confidence"]
        total_importance += importance
        chosen_key = choice_key(parsed_resp["choice"])
        reason_adjust = _reason_choice_adjustments(parsed_resp["reason"]) if chosen_key in OPTIONS else None

        per_role_scores = {}
        role_weights = ROLE_METRIC_WEIGHTS.get(role, ROLE_METRIC_WEIGHTS["None"])
        for opt in OPTIONS:
            u = utility_from_profile(role, profiles, opt, metric_weights=metric_weights)
            if opt == chosen_key and reason_adjust is not None:
                u += reason_adjust["utility_delta"]
                u -= metric_weights["collateral_risk"] * role_weights["collateral_risk"] * reason_adjust["collateral_delta"]
                u -= metric_weights["exposure_risk"] * role_weights["exposure_risk"] * reason_adjust["exposure_delta"]
                u -= metric_weights["catastrophic_risk"] * role_weights["catastrophic_risk"] * reason_adjust["catastrophic_delta"]
            score[opt] += importance * confidence * u

            opt_risk = profiles["catastrophic_risk"][opt]
            if opt == chosen_key and reason_adjust is not None:
                opt_risk = _clamp(opt_risk + reason_adjust["catastrophic_delta"])
            avg_risk[opt] += importance * opt_risk
            per_role_scores[opt] = u

        if chosen_key in OPTIONS:
            endorse_bonus = 0.10 + 0.06 * confidence
            endorsement[chosen_key] += importance * endorse_bonus
            score[chosen_key] += importance * endorse_bonus
            support_count[chosen_key] += 1
            support_roles[chosen_key].add(role)
            support_weight[chosen_key] += importance * confidence
            support_conf_sum[chosen_key] += confidence
            if role == "SafetyOfficer" and chosen_key not in parsed_resp.get("vetoes", set()):
                safety_supported[chosen_key] = True
            if reason_adjust is not None:
                if safety_veto and reason_adjust["veto"] and role == "SafetyOfficer":
                    safety_vetoes.add(chosen_key)

        role_views[agent_name] = {
            "role": role,
            "choice": parsed_resp["choice"],
            "confidence": confidence,
            "reason": parsed_resp["reason"],
            "utilities": {opt: round(per_role_scores[opt], 4) for opt in OPTIONS},
        }

        if safety_veto and role == "SafetyOfficer":
            safety_vetoes |= set(parsed_resp["vetoes"])

    if total_importance <= 0:
        total_importance = 1.0
    for opt in OPTIONS:
        avg_risk[opt] = avg_risk[opt] / total_importance

    consensus_bonus = {opt: 0.0 for opt in OPTIONS}
    support_summary = {}
    for opt in OPTIONS:
        cnt = support_count[opt]
        if cnt <= 0:
            support_summary[opt] = {
                "count": 0,
                "roles": [],
                "avg_confidence": 0.0,
                "support_weight": 0.0,
                "safety_supported": False,
            }
            continue
        avg_conf = support_conf_sum[opt] / max(cnt, 1)
        role_diversity = len(support_roles[opt])
        normalized_support = min(1.0, support_weight[opt] / max(total_importance, 1e-6))
        bonus = 0.0
        if cnt >= 2:
            bonus += 0.10 * avg_conf
        if role_diversity >= 2:
            bonus += 0.08 * avg_conf
        if cnt >= 3:
            bonus += 0.10 * avg_conf
        if safety_supported[opt]:
            bonus += 0.05 * avg_conf
        bonus *= max(0.65, normalized_support)
        score[opt] += bonus
        consensus_bonus[opt] = round(bonus, 4)
        support_summary[opt] = {
            "count": int(cnt),
            "roles": sorted(list(support_roles[opt])),
            "avg_confidence": round(avg_conf, 4),
            "support_weight": round(support_weight[opt], 4),
            "safety_supported": bool(safety_supported[opt]),
        }

    debate_bonus = {opt: 0.0 for opt in OPTIONS}
    debate_penalty = {opt: 0.0 for opt in OPTIONS}
    if debate_mode:
        debate_bonus, debate_penalty = _debate_mode_adjustments(
            sample,
            option_texts,
            score,
            avg_risk,
            support_count,
            support_roles,
            support_conf_sum,
        )

    if safety_veto:
        feasible = [opt for opt in OPTIONS if avg_risk[opt] <= epsilon and opt not in safety_vetoes]
    else:
        feasible = [opt for opt in OPTIONS if avg_risk[opt] <= epsilon]
    fallback_reason = None
    if not feasible:
        feasible = [opt for opt in OPTIONS if opt not in safety_vetoes] if safety_veto else list(OPTIONS)
        fallback_reason = "no_option_satisfies_epsilon"
    if not feasible:
        feasible = list(OPTIONS)
        fallback_reason = "all_options_vetoed"

    ranked = _rank_feasible_options(feasible, score, avg_risk, endorsement, vote_key, fallback_reason)
    if debate_mode:
        ranked = _promote_debate_consensus_choice(
            ranked,
            score,
            avg_risk,
            support_count,
            support_roles,
            option_texts,
            sample,
        )
    best = ranked[0]

    diagnostics = {
        "mode": "decision",
        "epsilon": epsilon,
        "feasible": feasible,
        "fallback_reason": fallback_reason,
        "vote_answer": vote_answer,
        "decision_differs_from_vote": normalize_choice(best) != normalize_choice(vote_answer),
        "vote_counts": vote_diag.get("counts", {}),
        "aggregated_score": {opt: round(score[opt], 4) for opt in OPTIONS},
        "endorsement": {opt: round(endorsement[opt], 4) for opt in OPTIONS},
        "consensus_bonus": consensus_bonus,
        "debate_mode": bool(debate_mode),
        "debate_bonus": debate_bonus,
        "debate_penalty": debate_penalty,
        "support_count": {opt: int(support_count[opt]) for opt in OPTIONS},
        "support_summary": support_summary,
        "avg_catastrophic_risk": {opt: round(avg_risk[opt], 4) for opt in OPTIONS},
        "profiles": deepcopy(profiles),
        "profile_source": profile_meta["profile_source"],
        "metric_weights": metric_weights,
        "safety_vetoes": sorted(safety_vetoes),
        "role_views": role_views,
        "ranked_options": ranked,
    }
    return final_answers, normalize_choice(best), diagnostics
