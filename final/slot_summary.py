"""
Slot-based summarization (期末进阶 #5).

Replaces the midterm's naive 'pick fact-shaped sentences' compressor with a
structured key-value extractor over a small ontology of common slots:

  name, profession, age, location, hometown, allergy, diet, family,
  pet, vehicle, birthday, password, salary, weight, deadline, destination,
  habit, schedule

For each slot we run a tiny rule extractor (regex + cue tokens) over the
oldest K messages, keep the LATEST extracted value per slot, and emit a
canonical "用户XXX: YYY" summary string. This gives a single high-density
entry that an LLM (or our mock LLM) can use cleanly.

This is structurally what an LLM summary would produce, but is rule-based
so it stays deterministic. In production this would be the call site for a
real LLM summarizer.
"""

import re
from typing import List, Dict, Optional


SLOT_PATTERNS = {
    "name":         [r"我叫([\u4e00-\u9fff]{2,4})", r"我是([\u4e00-\u9fff]{2,4})[，,]"],
    "profession":   [r"我是.{0,4}((?:[\u4e00-\u9fff]+)(?:研究员|经理|师|工程师|学生|医生|律师|教师|程序员))",
                     r"职业是([\u4e00-\u9fff]{2,8})"],
    "age":          [r"我(?:今年)?(\d{1,3})岁"],
    "location":     [r"住在([\u4e00-\u9fff]{2,8})", r"在([\u4e00-\u9fff]{2,6})(?:工作|生活)"],
    "hometown":     [r"老家(?:在|是)([\u4e00-\u9fff]{2,8})"],
    "allergy":      [r"对([\u4e00-\u9fff]{1,8})过敏", r"([\u4e00-\u9fff]{1,8})不耐受"],
    "diet":         [r"(素食|纯素|无糖|低盐|生酮|减脂|减肥)"],
    "family_kids":  [r"(\d+)个孩子", r"二孩", r"二胎"],
    "pet":          [r"([\u4e00-\u9fff]{1,4})是?(?:我家)?(?:的)?(柴犬|金毛|哈士奇|猫|狗|拉布拉多|泰迪)"],
    "vehicle":      [r"车牌(?:号)?是?([沪京津苏粤渝][A-Z][·\.0-9A-Z]+)"],
    "birthday":     [r"(?:生日|出生)(?:是|在)(\d+月\d+号)"],
    "password":     [r"密码(?:改成?了|是)([A-Za-z0-9#@!&\u4e00-\u9fff]{4,20})"],
    "salary":       [r"月收入.{0,3}(\d+(?:\.\d+)?)万"],
    "weight":       [r"体重(\d+(?:\.\d+)?)公斤"],
    "deadline":     [r"截止(?:日)?(?:是|改成?)(\d+月\d+号)"],
    "destination":  [r"行程改了[，,].{0,5}去([\u4e00-\u9fff]{2,8})",
                     r"去([\u4e00-\u9fff]{2,8})(?:旅游|玩|出差)"],
    "habit_coffee": [r"每天.{0,3}喝(\d+)杯", r"每天最多(\d+)杯"],
}


def extract_slots_from_message(msg: str) -> Dict[str, str]:
    """Return any slots successfully extracted from a single message."""
    found = {}
    for slot, patterns in SLOT_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, msg)
            if m:
                found[slot] = m.group(1) if m.groups() else m.group(0)
                break
    return found


def slot_summary(messages: List[str]) -> str:
    """
    Extract slots from each message; for each slot keep the LATEST value.
    (Latest wins handles updates and contradictions correctly.)
    Return a single canonical 用户档案 string.
    """
    slots: Dict[str, str] = {}
    for msg in messages:
        new = extract_slots_from_message(msg)
        slots.update(new)  # later overwrites earlier
    if not slots:
        return ""
    lines = ["[用户档案摘要]"]
    label = {
        "name": "姓名", "profession": "职业", "age": "年龄", "location": "居住地",
        "hometown": "老家", "allergy": "过敏", "diet": "饮食", "family_kids": "孩子数",
        "pet": "宠物", "vehicle": "车牌", "birthday": "生日", "password": "密码",
        "salary": "月收入(万)", "weight": "体重(公斤)", "deadline": "截止日期",
        "destination": "出行目的地", "habit_coffee": "每日咖啡杯数",
    }
    for slot, val in slots.items():
        lab = label.get(slot, slot)
        lines.append(f"- {lab}: {val}")
    return "\n".join(lines)


if __name__ == "__main__":
    msgs = [
        "我叫林清雨，是一名上海交大的计算机研三学生。",
        "我对宠物毛过敏。",
        "我家月收入大概4万。",
        "其实月收入也就1万出头，刚才说错了。",
        "我体重78公斤。",
        "我体重68公斤才对。",
        "我女儿明年要上小学一年级。",
        "更正，女儿明年上的是幼儿园中班。",
        "我下个月去日本旅游。",
        "行程改了，下个月去韩国。",
    ]
    summary = slot_summary(msgs)
    print(summary)
