#!/usr/bin/env python3
import os
import re
import json
import argparse
import pickle
from opencc import OpenCC
cc_t2s = OpenCC("t2s")
from typing import List, Set, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import random
import numpy as np
from collections import Counter
import unicodedata
from collections import defaultdict
from pypinyin import lazy_pinyin, Style
from transformers import AutoTokenizer
import Levenshtein

_DIGIT_MAP = {
    "1": "一", "2": "二", "3": "三", "4": "四",
    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"}
_ID_MAP = {
    "A": "1", "B": "2", "C": "3", "D": "4", "E": "5",
    "speaker_A": "1", "speaker_B": "2", "speaker_C": "3", "speaker_D": "4", "speaker_E": "5",
    "speaker_1": "1", "speaker_2": "2", "speaker_3": "3", "speaker_4": "4", "speaker_5": "5",
    "S1": "1", "S2": "2", "S3": "3", "S4": "4", "S5": "5"}
VALID_LABELS_ZH = {"中性", "喜悦", "信任", "害怕", "惊讶", "难过", "厌恶", "生气", "期待", "紧张", "不确定"}
VALID_LABELS_EN = {"neutral", "happy", "trust", "fear", "surprise","sadness", "disgust", "anger", "anticipation", "tension", "uncertain"}
VALID_AGE_ZH = {"儿童", "青年", "中年", "中老年", "老年", "不确定"}
VALID_AGE_EN = {"child", "teenager", "adult", "middle-aged", "elderly", "uncertain"}
VALID_GENDER_ZH = {"男", "女", "不确定"}
VALID_GENDER_EN = {"male", "female", "uncertain"}
SIM_THRESHOLDS = {"zh": 0.50, "en": 0.65}


def find_all_files(rttm_path: str) -> dict:
    """
    给定 rttm_path，查找与 basename 匹配的文件。
    """
    rttm_dir = os.path.dirname(rttm_path)
    parent_dir = os.path.dirname(rttm_dir)
    basename = os.path.splitext(os.path.basename(rttm_path))[0]
    film_name = os.path.basename(os.path.dirname(parent_dir))
    result = {
        "basename": basename, 
        "mp4":None, 
        "wav": None, 
        "tokens": None,
        "vocals": None, 
        "instrumental": None, 
        "srt": None, 
        "cot_wav": None, 
        "rttm": rttm_path,
        "embs_video": None,
        "embs_wav": None,
        "parent_dir": parent_dir,
        "film": film_name
    }
    for ext in ("mp4", "wav","srt"):
        p = os.path.join(parent_dir, "clipped", f"{basename}.{ext}")
        if os.path.exists(p):
            result[ext] = p
            
    p = os.path.join(parent_dir, "cot_wav", f"{basename}.json")
    if os.path.exists(p):
        result["cot_wav"] = p
    p = os.path.join(parent_dir, "tokens", f"{basename}.npy")
    if os.path.exists(p):
        result["tokens"] = p
    for subdir in ("vocals", "instrumental"):
        p = os.path.join(parent_dir, subdir, f"{basename}.wav")
        if os.path.exists(p):
            result[subdir] = p
            
    for subdir in ("embs_video", "embs_wav"):
        p = os.path.join(parent_dir, subdir, f"{basename}.pkl")
        if os.path.exists(p):
            result[subdir] = p
    return result


def find_rttm_files(root_dir: str) -> List[str]:
    rttm_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        # 找到名为 rttm 的目录
        if os.path.basename(dirpath).lower() == "rttm":
            for fn in filenames:
                if fn.lower().endswith(".rttm"):
                    rttm_paths.append(os.path.join(dirpath, fn))
    return rttm_paths

def is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF or
        0x3400 <= code <= 0x4DBF or
        0x20000 <= code <= 0x2A6DF or
        0x2A700 <= code <= 0x2B73F or
        0x2B740 <= code <= 0x2B81F or
        0x2B820 <= code <= 0x2CEAF or
        0xF900 <= code <= 0xFAFF or
        0x3000 <= code <= 0x303F
    )

def count_char_types(text: str) -> dict:
    """
    返回字符统计：
      - cjk: 被 is_cjk 识别的字符数（含中日韩统一表意、兼容区、以及常用 CJK 标点）
      - ascii: ASCII 英文字母数（A-Za-z）
      - non_cjk_letters: 不是 ASCII 且不是 CJK 的字母（例如：西里尔、希腊、阿拉伯、日/韩字母等）
      - digits: 数字字符数
      - others: 其他（标点、控制字符、emoji 等）
    """
    cjk = ascii_letters = non_cjk_letters = digits = others = 0
    for ch in text:
        if is_cjk(ch):
            cjk += 1
        elif ch.isalpha() and ord(ch) < 128:
            ascii_letters += 1
        elif ch.isalpha():
            non_cjk_letters += 1
        elif ch.isdigit():
            digits += 1
        else:
            others += 1
    return {
        'cjk': cjk,
        'ascii': ascii_letters,
        'non_cjk_letters': non_cjk_letters,
        'digits': digits,
        'others': others
    }


def try_fix_foreign(text: str, lang: str) -> Tuple[str, bool]:
    changed = False
    if lang == "zh":
        # 1) "Speaker|ID|Character|Actor|Role N" (N: 1-9)
        def _speaker_repl(m):
            return "说话人" + _DIGIT_MAP.get(m.group(1), m.group(1))
        text, nsub = re.subn(r'(?i)(?:Speaker|ID|Character|Actor|Role)\s*([1-9])', _speaker_repl, text)
        if nsub > 0:
            changed = True
        # 2) "S1/S2/S3"
        text, nsub = re.subn(r'(?i)(?<![A-Za-z0-9])S([1-9])(?![A-Za-z0-9])', 
                lambda m: "说话人" + _DIGIT_MAP.get(m.group(1), m.group(1)), text)
        if nsub > 0:
            changed = True
        # 3) 单字母（A B C）
        def _letter_repl(m):
            ch = m.group(1).upper()
            idx = ord(ch) - ord('A') + 1
            if 1 <= idx <= 9:
                return _DIGIT_MAP[str(idx)]
            return m.group(0)
        text, nsub = re.subn(r'(?<![A-Za-z0-9])([A-Za-z])(?![A-Za-z0-9])', _letter_repl, text)
        if nsub > 0:
            changed = True
        # 4) 性别标注 "Female" / "Male"
        text, nsub = re.subn(r'(?i)(?:female)', "女性", text)
        if nsub > 0:
            changed = True
        text, nsub = re.subn(r'(?i)(?:male)', "男性", text)
        if nsub > 0:
            changed = True
        # 5) other "playful|playfully|subtle|subtly|abrupt" ...
        text, nsub = re.subn(r'\s*\b(?:playful|playfully|subtle|subtly|abrupt|initial|respectful|rhetorical|dismissive)\b(?:\s*的)?\s*', '', text, flags=re.IGNORECASE)
        if nsub > 0:
            changed = True
        # 6) 移除时间戳
        time_pattern = r'\s*(?:\(|（)\s*\d+(?:\.\d+)?(?:\s*[-–—]\s*\d+(?:\.\d+)?)?\s*s\s*(?:\)|）)'
        text, nsub = re.subn(time_pattern, '', text, flags=re.IGNORECASE)
        if nsub > 0:
            changed = True
    elif lang == "en":
        # 1) 常见非ASCII标点映射，补充了西语标点
        PUNCT_MAPPING = {
            '。': '.', '，': ',', '！': '!', '？': '?', '：': ':', '；': ';',
            '（': '(', '）': ')', '「': '"', '」': '"', '『': '"', '』': '"',
            '、': ',', '·': '·', '…': '...', '—': '-', '\u3000': ' ', '\xa0': ' ',
            '¡': '!', '¿': '?', '«': '"', '»': '"', '„': '"', '‚': ',', '‘': "'", '’': "'"
        }
        for non_ascii, ascii_punct in PUNCT_MAPPING.items():
            if non_ascii in text:
                text = text.replace(non_ascii, ascii_punct)
                changed = True
                
        # 2） 重音字母规范化，如 "Raúl" → "Raul", "José" → "Jose", "café" → "cafe"
        normalized = unicodedata.normalize('NFKD', text)
        without_accents = ''.join(
            c for c in normalized 
            if unicodedata.category(c) != 'Mn'  # 'Mn'=Nonspacing_Mark（重音符号）
        )
        ascii_text = ''.join(c if ord(c) < 128 else ' ' for c in without_accents)
        if ascii_text != text:
            text = ascii_text
            changed = True
            
        # 3) 合并多余空格（避免"word   word"）
        text, nsub= re.subn(r'\s+', ' ', text)
        if nsub > 0:
            changed = True
    if _contains_foreign(text, lang):
        changed = False
    return text, changed


def _contains_foreign(text: str, lang: str) -> bool:
    """
    当 ascii > 0 或 non_cjk_letters > 0 时视为含外语脚本。
    """
    if lang == "zh":
        stats = count_char_types(text)    
        return (stats['ascii'] > 0) or (stats['non_cjk_letters'] > 0)
    elif lang == "en":
        return any(ord(c) > 127 for c in text)
    else:
        raise ValueError(f"Unsupported language code: {lang}. Use 'zh' or 'en'.")

def _contains_traditional(text: str) -> bool:
    """
    尝试检测繁体：
    """
    if not text:
        return False
    converted = cc_t2s.convert(text)
    if converted != text:
        return converted, True
    return text, False

def _atomic_writeback(obj: dict, path: str) -> None:
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as tf:
            json.dump(obj, tf, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"[ERROR] 无法写回 {path}: {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            print(f"[ERROR] 临时文件移除失败 {tmp_path}")

def remove_punctuation(text):
    """去除文本中的标点符号"""
    if not text:
        return ""
    text = re.sub(r'\s+', '', text)
    punctuation = '，。、！？；：“”‘’《》【】（）「」….,!?;:\'"\\|<>[]()-+=*&%#@……&*'
    for punc in punctuation:
        text = text.replace(punc, '')
    return text

def calculate_text_similarity(text1, text2):
    """
    计算两个文本的编辑距离相似度（不考虑标点）
    """
    text1_clean = remove_punctuation(text1)
    text2_clean = remove_punctuation(text2)
    
    # Levenshtein 距离
    distance = Levenshtein.distance(text1_clean, text2_clean)
    
    # 相似度比
    max_len = max(len(text1_clean), len(text2_clean))
    if max_len == 0:
        similarity = 1.0
    else:
        similarity = 1 - (distance / max_len)
    return similarity
    
def parse_rttm_speakers(rttm_path: str) -> Set[str]:
    """
    解析 rttm 文件，返回说话人 id 的集合。
    """
    speakers: Set[str] = set()
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            parts = line.split()
            # RTTM: TYPE FILE CHANNEL START DURATION ORTHO SPKR-TYPE SPKR-NAME ...
            spk = parts[7].strip()
            if spk.isdigit():
                speakers.add(spk)
            else:
                raise ValueError(f"解析 RTTM 失败 {rttm_path}")
    return speakers

def parse_srt_text(srt_path):
    """
    解析 srt 文件，返回字幕文本。
    """
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return lines[2].strip()


def parse_rttm_dialogue(rttm_path: str, meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    解析 RTTM 文件，产生 dialogue 列表。
    """
    meta_map = {}
    for item in meta:
        spk_id = str(item.get("id"))
        meta_map[spk_id] = item
    dialogues = []
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            start = float(parts[3])
            duration = float(parts[4])
            spk = parts[7]
            meta_entry = meta_map.get(spk)
            gender = meta_entry.get("gender")
            age = meta_entry.get("age")
            timbre = meta_entry.get("timbre")
            dialogues.append({
                "start": start,
                "duration": duration,
                "spk": spk,
                "gender": gender,
                "age": age,
                "timbre": timbre
            })
    return dialogues


def parse_cot(cot_wav_path: str, lang: str) -> Tuple[Set[str], bool, dict]:
    """
    解析 cot_wav json文件，缓解幻觉问题。
    """
    speakers: Set[str] = set()
    try:
        with open(cot_wav_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        print(f"[ERROR] cot_wav 文件读取失败: {cot_wav_path} {str(e)}")
        os.remove(cot_wav_path)
        return speakers, False, None
        
    changed = False
    
    label = obj.get("label", "")
    confidence = obj.get("confidence", -1.0)
    text = obj.get("text", "")
    clue = obj.get("clue", "")
    if clue == "" or text == "":
        print(f"[WARNING] cot_wav text/clue 为空 {cot_wav_path}")
        os.remove(cot_wav_path)
        return speakers, False, obj
    if not isinstance(clue, str) or not isinstance(text, str):
        print(f"[WARNING] cot_wav text 格式有误 {cot_wav_path}")
        os.remove(cot_wav_path)
        return speakers, False, obj
    
    # 外语脚本检测和修正（ASCII 或 非CJK字母）
    text_foreign = _contains_foreign(text, lang)
    clue_foreign = _contains_foreign(clue, lang)
    if text_foreign:
        new_text, changed_text = try_fix_foreign(text, lang)
        if changed_text:
            obj["text"] = new_text
            print(f"[INFO] {cot_wav_path} text 外文已修复: <{text}> <{new_text}>")
            text = new_text
            changed = True
        else:
            print(f"[WARNING] {cot_wav_path} text 含外文且无法修复")
            os.remove(cot_wav_path)
            return speakers, False, obj
    if clue_foreign:
        new_clue, changed_clue = try_fix_foreign(clue, lang)
        if changed_clue:
            obj["clue"] = new_clue
            print(f"[INFO] {cot_wav_path} clue 外文已修复: <{clue}> <{new_clue}>")
            clue = new_clue
            changed = True
        else:
            print(f"[WARNING] {cot_wav_path} clue 含外文且无法修复")
            os.remove(cot_wav_path)
            return speakers, False, obj
    # 繁体检测和修正
    if lang == "zh":
        new_text, mod1 = _contains_traditional(text)
        new_clue, mod2 = _contains_traditional(clue)
        if mod1:
            print(f"[INFO] {cot_wav_path} text 中文繁体已修复: <{text}> <{new_text}>")
            obj["text"] = new_text
            changed = True
        if mod2:
            print(f"[INFO] {cot_wav_path} clue 中文繁体已修复: <{clue}> <{new_clue}>")
            obj["clue"] = new_clue
            changed = True
            
    if lang == "zh" and label not in VALID_LABELS_ZH:
        print(f"[INFO] {cot_wav_path} label 异常 {label}, set to 不确定")
        obj['label'] = "不确定"
        changed = True
    if lang == "en":
        label_clean = label.strip().lower()
        if label_clean not in VALID_LABELS_EN:
            print(f"[INFO] {cot_wav_path} label 异常 {label}, set to uncertain")
            obj['label'] = "uncertain"
            changed = True
        elif label != label_clean:
            print(f"[INFO] {cot_wav_path} label 修复: <{label}> <{label_clean}>")
            obj['label'] = label_clean
            changed = True
                
    if not (0.0 <= confidence <= 1.0):
        print(f"[WARNING] {cot_wav_path} confidence 异常 {confidence}")
        os.remove(cot_wav_path)
        return speakers, False, obj
    
    spks = obj.get("speakers", [])
    for spk in spks[:]:
        spk_id = str(spk.get("id", ""))
        if spk_id.isdigit():
            speakers.add(spk_id)
        elif spk_id.lower() in ("background", "background_music"):
            spks.remove(spk)
            changed = True
            print(f"[INFO] {cot_wav_path} 删除 background 说话人")
            continue
        else:
            mapped = _ID_MAP.get(spk_id)
            if mapped is not None:
                spk['id'] = mapped
                speakers.add(mapped)
                changed = True
                print(f"[INFO] {cot_wav_path} 说话人 id 已修复: <{spk_id}> <{mapped}>")
            else:
                print(f"[WARNING] {cot_wav_path} 说话人 id 异常 {spk_id}")
                os.remove(cot_wav_path)
                return speakers, False, obj
        age = spk.get("age", "")
        if lang == "zh" and age not in VALID_AGE_ZH:
            print(f"[INFO] {cot_wav_path} 说话人年龄段异常 {age}, set to 不确定")
            spk['age'] = "不确定"
            changed = True
        if lang == "en":
            age_clean = age.strip().lower()
            if age_clean not in VALID_AGE_EN:
                print(f"[INFO] {cot_wav_path} 说话人年龄段异常 {age}, set to uncertain")
                spk['age'] = "uncertain"
                changed = True
            elif age != age_clean:
                print(f"[INFO] {cot_wav_path} 说话人年龄段修复: <{age}> <{age_clean}>")
                spk['age'] = age_clean
                changed = True
                
        gender = spk.get("gender", "")
        if lang == "zh" and gender not in VALID_GENDER_ZH:
            print(f"[INFO] {cot_wav_path} 说话人性别异常 {gender}, set to 不确定")
            spk['gender'] = "不确定"
            changed = True
        if lang == "en":
            gender_clean = gender.strip().lower()
            if gender_clean not in VALID_GENDER_EN:
                print(f"[INFO] {cot_wav_path} 说话人性别异常 {gender}, set to uncertain")
                spk['gender'] = "uncertain"
                changed = True
            elif gender != gender_clean:
                print(f"[INFO] {cot_wav_path} 说话人性别修复: <{gender}> <{gender_clean}>")
                spk['gender'] = gender_clean
                changed = True
        timbre = spk.get("timbre", "")
        if timbre == "":
            print(f"[WARNING] {cot_wav_path} 说话人角色描述为空 {spk_id} {timbre}")
            os.remove(cot_wav_path)
            return speakers, False, obj
        if _contains_foreign(timbre, lang):
            print(f"[WARNING] cot_wav timbre 含外文 {cot_wav_path} {spk_id} {timbre}")
            os.remove(cot_wav_path)
            return speakers, False, obj
        if lang == "zh":
            new_timbre, mod3 = _contains_traditional(timbre)
            if mod3:
                print(f"[INFO] {cot_wav_path} timbre 繁体已修复: <{timbre}> <{new_timbre}>")
                spk["timbre"] = new_timbre
                changed = True
    obj['speakers'] = spks
    if changed:
        _atomic_writeback(obj, cot_wav_path)
    return speakers, True, obj



def _extract_emotion_label(emotion_content):
    """
    emotion_content 格式示例: "喜悦 0.92"
    """
    match = re.search(r'<\|startofemo\|>\s*(.*?)\s*<\|endofemo\|>', emotion_content)
    clean_content = match.group(1).strip() if match else emotion_content.strip()
    parts = clean_content.split(maxsplit=1)
    label = parts[0].strip()
    return label

def _split_timbre(timbre):
    if not timbre or not isinstance(timbre, str):
        return []
    # 替换所有分隔符为空格
    normalized = re.sub(r'[、，,；;\s]+', ' ', timbre.strip())
    words = [w.strip() for w in normalized.split() if w.strip()]
    return words

def compute_and_save_film_stats(film_record: Dict[str, list], output_dir: str):
    stats = {}
    overall_emo = Counter()
    overall_age = Counter()
    overall_gender = Counter()
    overall_timbre = Counter()
    overall_type = Counter()
    overall_samples = 0
    overall_text_length = 0
    overall_clue_length = 0
    overall_speech_length = 0
    overall_unique_speakers = 0
    for film, recs in film_record.items():
        total_samples = len(recs)
        overall_samples += total_samples
        total_text_length = 0
        total_clue_length = 0
        total_speech_length = 0
        emo_counter = Counter()
        age_counter = Counter()
        gender_counter = Counter()
        timbre_counter = Counter()
        type_counter = Counter()
        unique_speakers = 0

        for rec in recs:
            # type
            sample_type = rec.get("type")
            type_counter[sample_type] += 1
            overall_type[sample_type] += 1
            
            # emotion
            emo_content = None
            for m in rec.get("messages", []):
                if m.get("role") == "emotion":
                    emo_content = m.get("content")
                    break
            label = _extract_emotion_label(emo_content)
            emo_counter[label] += 1
            overall_emo[label] += 1

            # 统计每部剧的不重复说话人年龄和性别
            dialogue = None
            for m in rec.get("messages", []):
                if m.get("role") == "dialogue":
                    dialogue = m.get("content")
                    break
            seen_spks = set()
            for turn in dialogue:
                spk = str(turn.get("spk"))
                age = turn.get("age")
                gender = turn.get("gender")
                timbre = turn.get("timbre")
                if spk not in seen_spks:
                    seen_spks.add(spk)
                    age_counter[age] += 1
                    overall_age[age] += 1
                    gender_counter[gender] += 1
                    overall_gender[gender] += 1
                    unique_speakers += 1
                    overall_unique_speakers += 1
                    timbre_words = _split_timbre(timbre)
                    for word in timbre_words:
                        timbre_counter[word] += 1
                        overall_timbre[word] += 1
            
            # 统计每部剧平均样本的语音/文本/线索的tokens长度
            text_length = rec.get("text_length")
            clue_length = rec.get("clue_length")
            speech_length = rec.get("speech_length")
            total_text_length += text_length
            overall_text_length += text_length
            total_clue_length += clue_length
            overall_clue_length += clue_length
            total_speech_length += speech_length
            overall_speech_length += speech_length
        
        avg_text_length = round(total_text_length / total_samples, 2) if total_samples > 0 else 0.0
        avg_clue_length = round(total_clue_length / total_samples, 2) if total_samples > 0 else 0.0
        avg_speech_length = round(total_speech_length / total_samples, 2) if total_samples > 0 else 0.0
        
        type_stats = {}
        for type_label, cnt in type_counter.items():
            type_stats[type_label] = {
                "count": cnt,
                "percent": round(cnt * 100.0 / total_samples, 2) if total_samples > 0 else 0.0
            }
        emo_stats = {}
        for emo_label, cnt in emo_counter.items():
            emo_stats[emo_label] = {
                "count": cnt,
                "percent": round(cnt * 100.0 / total_samples, 2) if total_samples > 0 else 0.0
            }
        denom = unique_speakers if unique_speakers > 0 else max(1, total_samples)
        age_stats = {}
        for age_label, cnt in age_counter.items():
            age_stats[age_label] = {
                "count": cnt,
                "percent": round(cnt * 100.0 / denom, 2)
            }
        gender_stats = {}
        for gender_label, cnt in gender_counter.items():
            gender_stats[gender_label] = {
                "count": cnt,
                "percent": round(cnt * 100.0 / denom, 2)
            }
        timbre_stats = [{"word": word, "count": count} for word, count in timbre_counter.most_common(30)]

        stats[film] = {
            "total_samples": total_samples,
            "total_speakers_counted": unique_speakers,
            "type_distribution": type_stats,
            "emotion_distribution": emo_stats,
            "age_distribution": age_stats,
            "gender_distribution": gender_stats,
            "avg_text_length": avg_text_length,
            "avg_clue_length": avg_clue_length,
            "avg_speech_length": avg_speech_length,
            "timbre_top30": timbre_stats
        }
    
    overall = {}
    overall["total_samples"] = overall_samples
    overall["avg_text_length"] = round(overall_text_length / overall_samples, 2) if overall_samples > 0 else 0.0
    overall["avg_clue_length"] = round(overall_clue_length / overall_samples, 2) if overall_samples > 0 else 0.0
    overall["avg_speech_length"] = round(overall_speech_length / overall_samples, 2) if overall_samples > 0 else 0.0
    overall["total_unique_speakers_counted"] = overall_unique_speakers
    overall["sample_type_distribution"] = {k: {"count": v, "percent": round(v * 100.0 / max(1, overall_samples), 2)} for k, v in overall_type.items()}
    overall["emotion_distribution"] = {k: {"count": v, "percent": round(v * 100.0 / max(1, overall_samples), 2)} for k, v in overall_emo.items()}
    overall["age_distribution"] = {k: {"count": v, "percent": round(v * 100.0 / max(1, overall_unique_speakers), 2)} for k, v in overall_age.items()}
    overall["gender_distribution"] = {k: {"count": v, "percent": round(v * 100.0 / max(1, overall_unique_speakers), 2)} for k, v in overall_gender.items()}
    overall["timbre_top30"] = [{"word": word, "count": count} for word, count in overall_timbre.most_common(30)]
    # 保存到文件
    out_path = os.path.join(output_dir, "film_stats_per_film.json")
    with open(out_path, "w", encoding="utf-8") as fo:
        json.dump(stats, fo, ensure_ascii=False, indent=2)
    print(f"[INFO] 已保存每部影视剧统计到: {out_path}")
    out_path = os.path.join(output_dir, "film_stats.json")
    with open(out_path, "w", encoding="utf-8") as fo:
        json.dump(overall, fo, ensure_ascii=False, indent=2)
    print(f"[INFO] 已保存全部统计数据到: {out_path}")



def process_single_rttm(rttm_path, lang, tokenizer):
    files = find_all_files(rttm_path)
    basename = files["basename"]
    film = files["film"]
    pinyin_film= ''.join(lazy_pinyin(film, style=Style.NORMAL))
    utt = f"{pinyin_film}_{basename}"
    missing_files = [key for key, value in files.items() if value is None]
    if missing_files:
        return {
            "basename": basename,
            "status": "skip",
            "reason": f"缺失文件: {', '.join(missing_files)}"
        }
    cot_spk, can_use, cot_obj = parse_cot(files["cot_wav"], lang)
    if not can_use:
        return {
            "basename": basename,
            "status": "skip",
            "reason": "cot内容异常"
        }
    # corrected 文本与 srt 文本编辑距离
    srt_text = parse_srt_text(files["srt"])
    cot_text = cot_obj.get("text")
    text_sim = calculate_text_similarity(cot_text, srt_text)
    threshold = SIM_THRESHOLDS.get(lang)
    
    if text_sim < threshold:
        print(f"[WARNING] {utt} cot文本与srt文本相似度过低: {text_sim:.2f} < {threshold}")
        return {
            "basename": basename,
            "status": "skip",
            "reason": f"cot文本与srt文本相似度过低: {text_sim:.2f}"
        }
    # 解析 rttm 说话人
    rttm_spk = parse_rttm_speakers(rttm_path)
    is_equal = rttm_spk == cot_spk
    rttm_is_subset = (rttm_spk.issubset(cot_spk) and not is_equal)
    cot_is_subset = (cot_spk.issubset(rttm_spk) and not is_equal)
    result = {
        "basename": basename,
        "film": film,
        "status": "success",
        "reason": "",
        "record": {},
        "sample_type": None,
        "rttm_spk_count": len(rttm_spk),
        "cot_spk_count": len(cot_spk),
        "is_equal": is_equal,
        "rttm_is_subset": rttm_is_subset,
        "cot_is_subset": cot_is_subset
    }
    if is_equal:
        speech_tokens_file = files.get("tokens")
        speech_tokens = np.load(speech_tokens_file)
        speech_length = len(speech_tokens)
        del speech_tokens

        visual_embs_file = files.get("embs_video")
        with open(visual_embs_file, 'rb') as f:
            video_obj = pickle.load(f)
        frameI = video_obj['frameI']
        frameI_count = len(frameI)
        faceI = video_obj['faceI']
        faceI_count = len(faceI)
        # 判断样本类型
        if frameI_count == 0 and faceI_count >= 0 and speech_length>0:
            result["sample_type"] = "旁白"
        elif frameI_count > 0 and faceI_count > 0 and faceI_count >= frameI_count and speech_length>0:
            if result["cot_spk_count"] == 1:
                result["sample_type"] = "独白"
            elif result["cot_spk_count"] == 2:
                result["sample_type"] = "对话"
            elif result["cot_spk_count"] > 2:
                result["sample_type"] = "多人"
        else:
            result["sample_type"] = "其他"
        dialogue = parse_rttm_dialogue(rttm_path, cot_obj.get("speakers"))
        record = {
            "messages": [
                {"role": "text", "content": cot_obj.get("text")},
                {"role": "token", "content": files.get("tokens")},
                {"role": "vocal", "content": files.get("vocals")},
                {"role": "instrumental", "content": files.get("instrumental")},
                {"role": "video", "content": files.get("mp4")},
                {"role": "face", "content": files.get("embs_video")},
                {"role": "embswav", "content": files.get("embs_wav")},
                {"role": "dialogue", "content": dialogue},
                {"role": "clue", "content": cot_obj.get("clue").strip()},
                {"role": "emotion", "content": "{} {}".format(cot_obj.get("label").strip(), cot_obj.get("confidence"))},
            ],
            "utt": utt,
            "type": result["sample_type"],
            "source": lang,
            "task": "VTTS",
            "text_length": len(tokenizer.encode(cot_obj.get("text"))),
            "clue_length": len(tokenizer.encode(cot_obj.get("clue"))),
            "speech_length": speech_length
        }
        result["record"] = record
    else:
        result["sample_type"] = "不等"
    return result


def batch_process(root_zh: str, root_en: str, output_dir: str, tokenizer_path: str, workers: int, save: bool):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)        
    rttm_entries = []  # (rttm_path, lang)
    if root_zh and os.path.isdir(root_zh):
        zh_files = find_rttm_files(root_zh)
        print(f"[INFO] 从中文目录 {root_zh} 找到 {len(zh_files)} 个RTTM文件")
        rttm_entries.extend([(f, "zh") for f in zh_files])
    if root_en and os.path.isdir(root_en):
        en_files = find_rttm_files(root_en)
        print(f"[INFO] 从英文目录 {root_en} 找到 {len(en_files)} 个RTTM文件")
        rttm_entries.extend([(f, "en") for f in en_files])
    if not rttm_entries:
        print(f"[ERROR] 未找到任何RTTM文件")
        return
    print(f"[INFO] 启动 {workers} workers 处理...")
    
    total_count = len(rttm_entries)
    skip_count = 0
    not_sim = 0
    sucess_count = 0
    rttm_subset_count = 0
    cot_subset_count = 0
    equal_count = 0
    pangbai = 0
    dubai = 0
    duihua = 0
    duoren = 0
    budeng = 0
    other = 0
    film_record = defaultdict(list)
                    
    with ThreadPoolExecutor(max_workers=workers) as exe:
        func = partial(process_single_rttm, tokenizer=tokenizer)
        futures = {
            exe.submit(func, rttm_path, lang): (rttm_path, lang)
            for (rttm_path, lang) in rttm_entries
        }
        for i, fut in enumerate(as_completed(futures)):
            try:
                result = fut.result(timeout=120)  
            except Exception as e:
                print(f"[ERROR] {str(e)}")
                skip_count += 1
                continue
            
            if result["status"] == "skip":
                skip_count += 1
                if "本相似度过低" in result["reason"]:
                    not_sim += 1
            else:
                sucess_count += 1
                if result["rttm_is_subset"]:
                    rttm_subset_count += 1
                if result["cot_is_subset"]:
                    cot_subset_count += 1
                if result["is_equal"]:
                    equal_count += 1
                    film_record[result["film"]].append(result["record"])
                    
                    
                if result["sample_type"] == "旁白":
                    pangbai += 1
                elif result["sample_type"] == "独白":
                    dubai += 1
                elif result["sample_type"] == "对话":
                    duihua += 1
                elif result["sample_type"] == "多人":
                    duoren += 1
                elif result["sample_type"] == "不等":
                    budeng +=1
                else:
                    other += 1
    # save results
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")
    if save:        
        with open(train_path, "w", encoding="utf-8") as ftrain, open(test_path, "w", encoding="utf-8") as ftest:
            for _, recs in film_record.items():
                test_indices = random.sample(range(len(recs)), 5) # 每部电影随机选取5个样本作为测试集
                test_set = set(test_indices)
                for idx, rec in enumerate(recs):
                    if idx in test_set:
                        ftest.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    else:
                        ftrain.write(json.dumps(rec, ensure_ascii=False) + "\n")
    compute_and_save_film_stats(film_record, output_dir)
    
                
    print("=== Summary ===")
    print(f"Total RTTM files: {total_count}")
    print(f"Successful processed: {sucess_count}")
    print(f"Skip: {skip_count}")
    print(f"CoT 与 SRT 文本偏差过大: {not_sim}") 
    print(f"RTTM ⊂ COT count: {rttm_subset_count}")
    print(f"COT ⊂ RTTM count: {cot_subset_count}")
    print(f"RTTM == COT count: {equal_count}")
    print(f"旁白: {pangbai}")
    print(f"独白: {dubai}")
    print(f"对话: {duihua}")
    print(f"多人: {duoren}")
    print(f"不等: {budeng}")
    print(f"其他: {other}")
    if save: print(f"Train file: {train_path}; Test file: {test_path}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取最优数据集")
    parser.add_argument("--root_zh", default=None, help="中文数据集根目录")
    parser.add_argument("--root_en", default=None, help="英文数据集根目录")
    parser.add_argument("--out_dir", required=True, help="保存 train/test jsonl 的目录")
    parser.add_argument("--workers", type=int, default=120, help="并发 worker 数")
    parser.add_argument("--tokenizer_path", default="tokenizer/Qwen2-0.5B-CosyVoice-BlankEN", help="预测text token length 的tokenizer目录")
    parser.add_argument("--seed", type=int, default=100, help="随机种子")
    parser.add_argument("--save", action="store_true", help="是否划分测试集和训练集并保存")
    args = parser.parse_args()
    
    zh_exists = args.root_zh is not None and os.path.isdir(args.root_zh)
    en_exists = args.root_en is not None and os.path.isdir(args.root_en)
    if not zh_exists and not en_exists:
        print(f"[ERROR] 请在参数 root_zh 与 root_en 中提供正确的中英文数据路径: zh={args.root_zh}, en={args.root_en}")
        exit(2)
    if zh_exists:
        print(f"[INFO] 中文数据集: {args.root_zh}")
    if en_exists:
        print(f"[INFO] 英文数据集: {args.root_en}")
        
    print("[INFO] 生成数据集并统计..." if args.save else "[INFO] 仅进行双向验证修正文件并统计...")
    random.seed(args.seed)
    batch_process(args.root_zh, args.root_en, output_dir=args.out_dir, tokenizer_path = args.tokenizer_path, workers=args.workers, save=args.save)