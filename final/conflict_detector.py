"""
Conflict detector (期末进阶 #4).

When the top-K retrieved memory chunks contain factually contradictory
information (different numbers, different city names for the same topic,
etc.), this module flags it. The Agent can then trigger a clarification
question instead of blindly committing to one value.

Detection strategies (lightweight, deterministic):
  1. Numeric conflict: two retrieved chunks contain different numbers for
     the same NOUN cue (e.g., "10人" vs "12人", "78公斤" vs "68公斤").
  2. Categorical conflict: two retrieved chunks contain different values
     from a small closed set tied to the same slot (e.g., a city slot
     like {上海, 北京, 杭州, ...}).
  3. Negation conflict: one chunk asserts X and another negates X
     (e.g., "我喝啤酒" vs "我从不喝酒").
"""

import re
from typing import List, Dict, Tuple, Any, Optional

NUM_PATTERN = re.compile(r"(\d+(?:\.\d+)?)(?:公斤|kg|岁|杯|元|万|人|号|月|个)?")
CITY_LIST = ["北京", "上海", "杭州", "南京", "苏州", "广州", "深圳", "成都",
             "重庆", "天津", "西安", "武汉", "长沙", "青岛", "济南", "厦门",
             "福州", "大连", "沈阳", "哈尔滨", "三亚", "海口", "贵阳", "南宁",
             "兰州", "昆明", "拉萨", "乌鲁木齐", "呼和浩特", "石家庄"]
COUNTRIES = ["日本", "韩国", "美国", "英国", "法国", "德国", "意大利", "西班牙",
             "俄罗斯", "印度", "加拿大", "澳大利亚", "新西兰", "泰国", "越南",
             "马来西亚", "新加坡", "印尼"]
NEG_CUES = ["不", "没"]


def extract_numbers(text: str) -> List[str]:
    """Return all numeric tokens in the text (as raw strings)."""
    return NUM_PATTERN.findall(text)


def extract_cities(text: str) -> List[str]:
    return [c for c in CITY_LIST if c in text]


def extract_countries(text: str) -> List[str]:
    return [c for c in COUNTRIES if c in text]


def detect_numeric_conflict(chunks: List[str]) -> Optional[Dict]:
    """Two chunks contain different numbers attached to a shared keyword."""
    if len(chunks) < 2:
        return None
    sets = [set(extract_numbers(c)) for c in chunks]
    flat = set().union(*sets)
    if len(flat) < 2:
        return None
    # which chunks contain which numbers?
    distinct_pairs = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            ni, nj = sets[i], sets[j]
            # different numbers AND overlap-vocabulary on >=1 Chinese char (rough topic match)
            shared = set(re.findall(r"[\u4e00-\u9fff]", chunks[i])) & set(re.findall(r"[\u4e00-\u9fff]", chunks[j]))
            if (ni - nj or nj - ni) and len(shared) >= 2:
                distinct_pairs.append((i, j, ni, nj, shared))
    if distinct_pairs:
        i, j, ni, nj, shared = distinct_pairs[0]
        return {
            "type": "numeric",
            "chunks": (chunks[i], chunks[j]),
            "values": (sorted(ni), sorted(nj)),
            "shared_context_chars": "".join(sorted(shared))[:10],
        }
    return None


def detect_categorical_conflict(chunks: List[str]) -> Optional[Dict]:
    """Two chunks mention different cities/countries that look co-referential."""
    if len(chunks) < 2:
        return None
    for slot_name, extractor in [("city", extract_cities), ("country", extract_countries)]:
        per_chunk = [set(extractor(c)) for c in chunks]
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                ci, cj = per_chunk[i], per_chunk[j]
                if ci and cj and ci.isdisjoint(cj):
                    shared = set(re.findall(r"[\u4e00-\u9fff]", chunks[i])) & set(re.findall(r"[\u4e00-\u9fff]", chunks[j]))
                    if len(shared) >= 2:
                        return {
                            "type": f"categorical_{slot_name}",
                            "chunks": (chunks[i], chunks[j]),
                            "values": (sorted(ci), sorted(cj)),
                            "shared_context_chars": "".join(sorted(shared))[:10],
                        }
    return None


def detect_negation_conflict(chunks: List[str]) -> Optional[Dict]:
    """One chunk has a verb/noun phrase that another negates."""
    if len(chunks) < 2:
        return None
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            ci, cj = chunks[i], chunks[j]
            ci_neg = any(n in ci for n in NEG_CUES)
            cj_neg = any(n in cj for n in NEG_CUES)
            # exactly one of them is negated
            if ci_neg ^ cj_neg:
                shared = set(re.findall(r"[\u4e00-\u9fff]", ci)) & set(re.findall(r"[\u4e00-\u9fff]", cj))
                # require a moderate vocabulary overlap on Chinese chars
                if len(shared) >= 3:
                    return {
                        "type": "negation",
                        "chunks": (ci, cj),
                        "shared_context_chars": "".join(sorted(shared))[:10],
                    }
    return None


def detect_conflicts(retrieved: List[Tuple[str, float, Dict[str, Any]]]) -> List[Dict]:
    """Run all three detectors over the retrieved chunks."""
    chunks = [r[0] for r in retrieved]
    out = []
    for detector in (detect_numeric_conflict, detect_categorical_conflict, detect_negation_conflict):
        conf = detector(chunks)
        if conf:
            out.append(conf)
    return out


def format_clarification(conflict: Dict) -> str:
    """Produce a polite clarification utterance."""
    if conflict["type"] == "numeric":
        v1, v2 = conflict["values"]
        return (f"我注意到您之前提到的数字有出入（{','.join(v1)} vs {','.join(v2)}），"
                "请问以哪个为准？")
    if conflict["type"].startswith("categorical"):
        v1, v2 = conflict["values"]
        slot = conflict["type"].split("_", 1)[1]
        return f"我注意到您之前提到的{slot}存在不一致（{','.join(v1)} vs {','.join(v2)}），请问以哪个为准？"
    if conflict["type"] == "negation":
        return "我注意到您之前的描述存在前后矛盾，能否帮我确认一下最新的情况？"
    return "我注意到您之前的信息存在不一致，能否确认一下？"


if __name__ == "__main__":
    cases = [
        # numeric
        [("我体重78公斤，要减脂", 0.5, {}), ("我体重68公斤才对", 0.5, {})],
        # categorical-city
        [("我女朋友是上海人", 0.5, {}), ("她其实是北京人", 0.5, {})],
        # categorical-country
        [("我下个月去日本旅游", 0.5, {}), ("行程改了，下个月去韩国", 0.5, {})],
        # negation
        [("我最近开始喝啤酒", 0.5, {}), ("我从不喝酒", 0.5, {})],
        # no conflict
        [("我家wifi密码是Lin8520", 0.5, {}), ("妈妈生日是10月17号", 0.5, {})],
    ]
    for i, c in enumerate(cases):
        confs = detect_conflicts(c)
        print(f"\nCase {i+1}: {[h[0] for h in c]}")
        if confs:
            for cc in confs:
                print(f"  → conflict {cc['type']}: {cc.get('values') or cc.get('chunks')}")
                print(f"  → ask: {format_clarification(cc)}")
        else:
            print("  → no conflict detected")
