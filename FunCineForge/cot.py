#!/usr/bin/env python3
"""
用音视频通用 caption 大模型做 audio / video CoT分析和矫正。
我们依赖接口: requests url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    本工具推荐并采用 gemini-3-pro-preview model， 可自行更换其他多模态理解大模型如 Qwen 3.5， gemini 3 flash
    python cot.py --root_dir datasets/clean/zh --provider google --model gemini-3-pro-preview --api_key xxx --resume
可更换 url 依赖，你可自行参考 https://ai.google.dev/gemini-api/docs/structured-output ，并修改 call_dashscope_api 代码
from google import genai
client = genai.Client()
response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents=messages,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=1024)),
)
"""

import os
import sys
import base64
import json
import re
import argparse
from typing import List, Dict, Any, Tuple
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.exceptions import Timeout, ConnectionError, HTTPError
import time
import gc

# ------------------------------
# 构造 Prompt（音频 / 视频）
# ------------------------------
AUDIO_PROMPT_TEMPLATE_zh = """
你是一个专注于中文音频分析和纠错的专家，任务是使用辅助信息分析给定的 WAV 音频文件，辅助信息有 ASR 转录文本，和从RTTM中提取的带有说话人 id（1，2，3 等）的时间戳。
注意 ASR 转录文本会存在词汇错误和标点断句错误的情况，说话人 id 可能存在少标或多标，请你根据语音中实际的说话人音色来区分说话人 id，纠正后的 id 需要和 RTTM 的说话人 id 匹配。
任务：
1、确定音频中说话人的数量。
2、分析每个说话人的年龄段、性别和音色属性。
3、准确识别语音内容，认真参考提供的 ASR 转录文本，对 ASR 转录文本进行纠错，给出纠正后的文本和合理的中文标点。
4、根据语音内容和 RTTM 信息，理解音频的每个时间段内是哪个说话人在说话。
5、分析并总结音频的整体情感线索，和每个说话人的属性、语气和情感，并保存在 clue 字段中，线索用中文陈述，不描述背景音效，线索字数不超过100字。
输出要求：
1、text 为矫正后的转录文本带有合理的中文标点。
2、label 为 <中性、喜悦、信任、害怕、惊讶、难过、厌恶、生气、期待、紧张、不确定> 中一种，confidence 为 label 的置信度（0.0 到 1.0）。
3、speakers 包含以下对象的列表：
id 必须与原始 RTTM id 编号系统一致，但分配逻辑基于音频内容，正确的 id 序号要和 RTTM 的 id 编号一致；
age 为 <儿童，青年，中年，中老年，老年，不确定> 中一种；
gender 为<男，女，不确定>中一种；
timbre 为两三个描述音色的形容词（例如：深沉、柔和、磁性、疑虑）。
4、clue 是对每位说话人的属性、情感和语气的描述。
你必须严格按照下面的JSON格式输出，具体内容请根据实际音频填充，不能输出任何其他内容。
<answer>{"label": "紧张", "confidence": 0.7, "text": "哎呀，将军，将军，不可连累老夫啊！大丈夫生居天地之间，岂能郁郁久居人下！", "speakers": [{"id": "1", "age": "中年", "gender": "男", "timbre": "低沉、苍老"}, {"id": "2", "age": "青年", "gender": "男", "timbre": "高亢、有力、果断"}, ...], "clue": "两名角色对话，第一位中年男性角色情绪紧张，语气略带颤抖和哀求，表达对被牵连的恐惧。第二位角色语调变得激昂坚定，铿锵有力，充满对尊严和自由的强烈渴望。整体展现出从畏惧到反抗的情感转变。"}</answer>
"""

AUDIO_PROMPT_TEMPLATE_en = """
You are an expert specializing in English audio analysis and error correction. Your task is to analyze a given WAV audio file using auxiliary information: an ASR transcript and RTTM-extracted timestamps with speaker IDs (1, 2, 3, etc.).
Note: The ASR transcript may contain word errors and punctuation mistakes. Speaker IDs might be mislabeled (too many or too few). You must distinguish speaker IDs based on the actual vocal timbre in the audio and ensure corrected IDs match the RTTM IDs.
Tasks:
1. Determine the number of speakers in the audio.
2. Analyze each speaker's age group, gender, and timbre attributes.
3. Accurately identify the speech content. Refer to the provided ASR transcript to correct errors, providing the correct text and natural English punctuation.
4. Based on the audio content and RTTM info, determine which speaker is speaking in each time interval of the audio.
5. Analyze the overall emotion and each speaker's attributes, tone and emotion. Summarize the speakers' attributes and emotional clues in the "clue" field.
The "clue" must be in English, exclude background sound descriptions, and be under 150 words.
Output Requirements:
1. text: Corrected transcription with proper English punctuation.
2. label: Choose from <neutral, happy, trust, fear, surprise, sadness, disgust, anger, anticipation, tension, uncertain>. Confidence score for the label (0.0 to 1.0).
3. speakers: A list of objects containing:
id: The output speaker_id must be consistent with the original RTTM ID numbering system, but the allocation logic is based on the audio content;
age: <child, teenager, adult, middle-aged, elderly, uncertain>; 
gender: <male, female, uncertain>; 
timbre: 2-3 descriptive adjectives (e.g., deep, gentle, magnetic, doubtful).
clue: A summary description of each speaker's attributes, emotions, and tone.
You must strictly follow the dictionary template below to output the results. Only output the result in the style shown below, without any other additional output.
<answer>{"label": "surprise", "confidence": 0.8, "text": "Oh, that is absolutely wonderful news! I can't believe we actually won the championship!", "speakers": [{"id": "1", "age": "teenager", "gender": "male", "timbre": "bright, energetic"}, {"id": "2", "age": "middle-aged", "gender": "female", "timbre": "warm, resonant"}, ...], "clue": "A dialogue between two speakers. The first young male speaker expresses intense excitement and disbelief about a victory. The second middle-aged female speaker responds with a warm, supportive tone. The overall atmosphere is celebratory and uplifting."}</answer>
"""

VIDEO_PROMPT_TEMPLATE = """
"""


# ------------------------------
# 辅助函数
# ------------------------------
# 价格（美元 / 每百万 token）
_PRICE_TEXT_PER_M = 0.5 / 1_000_000
_PRICE_AUDIO_PER_M = 0.5 / 1_000_000
_PRICE_OUTPUT_PER_M = 1.0 / 1_000_000

def calculate_cost_from_usage(usage: Dict[str, Any]) -> Tuple[int, float]:
    """
    根据 usage 字段计算成本, 对缺失字段使用 0 作为默认值。
    """
    prompt_details = usage.get('prompt_tokens_details', {}) or {}
    audio_tokens = int(prompt_details.get('audio_tokens', 0) or 0)
    text_tokens = int(prompt_details.get('text_tokens', 0) or 0)

    completion_tokens = int(usage.get('completion_tokens', 0) or 0)
    think_tokens = int(usage.get('reasoning_tokens', 0) or 0)
    if completion_tokens == 0:
        completion_tokens = int((usage.get('completion_tokens_details', {}) or {}).get('text_tokens', 0) or 0)

    # 计算成本
    cost_input_audio = audio_tokens * _PRICE_AUDIO_PER_M
    cost_input_text = text_tokens * _PRICE_TEXT_PER_M
    cost_output = (completion_tokens + think_tokens) * _PRICE_OUTPUT_PER_M
    total_tokens = audio_tokens + text_tokens + completion_tokens + think_tokens
    total_cost = cost_input_audio + cost_input_text + cost_output

    return total_tokens, total_cost

def read_file_as_data_url(path: str, mime: str) -> str:
    with open(path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("utf-8")
    del b
    gc.collect()
    return f"data:{mime};base64,{b64}"

def parse_srt(srt_path: str) -> str:
    """
    读取 SRT 文件，返回第最后一行ASR text内容。
    """
    with open(srt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[-1] if lines else ""

def find_files_for_rttm(rttm_path):
    """
    给定 rttm_path，查找 sibling clipped 目录下与 basename 匹配的 wav/srt/mp4。
    """
    rttm_dir = os.path.dirname(rttm_path)
    basename = os.path.splitext(os.path.basename(rttm_path))[0]
    parent_dir = os.path.dirname(rttm_dir)
    result = {"wav": None, "srt": None, "basename": basename, "parent_dir": parent_dir}
    srt = os.path.join(parent_dir, "clipped", f"{basename}.srt")
    wav = os.path.join(parent_dir, "vocals", f"{basename}.wav")
    if os.path.exists(srt) and os.path.exists(wav):
        result["srt"] = srt
        result["wav"] = wav
    else:
        print(f"[WARNING] 未找到匹配的 clipped wav/srt: {wav}, {srt}")
    return result

def find_all_rttm_files(root_dir: str) -> List[str]:
    rttm_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        # 找到名为 rttm 的目录
        if os.path.basename(dirpath).lower() == "rttm":
            for fn in filenames:
                if fn.lower().endswith(".rttm"):
                    rttm_paths.append(os.path.join(dirpath, fn))
    return rttm_paths

def parse_rttm(rttm_path: str) -> List[Dict]:
    """
    解析 RTTM 文件为段列表。
    支持格式: SPEAKER 05_00_01_50_60 1 0.000 1.390 <NA> <NA> 1 <NA> <NA>
    """
    segments = []
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            start = float(parts[3])
            duration = float(parts[4])
            end = start + duration
            speaker = parts[7]

            segments.append({
                "start": start,
                "end": end,
                "speaker": speaker,
            })
    # 按时间排序（可选）
    segments.sort(key=lambda x: (x["start"], x["end"]))
    # 检测重叠
    for i in range(len(segments) - 1):
        if segments[i]["end"] > segments[i + 1]["start"]:
            print(f"[WARNING] 检测到重叠片段: {rttm_path}")
            break
    return segments


def call_dashscope_api(api_key: str, provider: str, model_name: str, messages: List[Dict],
                       thinking_budget: int = 1024, max_tokens: int = 16000, timeout: int = 200, max_retries: int = 5) -> Dict:
    """
    调用 DashScope API，失败时自动重试
    """
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": api_key}
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {
                "includeThoughts": "true",
                "thinkingBudget": thinking_budget
            }
        },
        "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
        "dashscope_extend_params": {"provider": provider}
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
            if resp.status_code in [400, 429, 500, 502, 503, 504]:
                retry_after = int(resp.headers.get("Retry-After", 2 ** attempt))
                wait = max(retry_after, 0.5 * (2 ** attempt))  # 指数退避
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(0.3) # 避免连续请求触发限流
            return resp.json()
        except (Timeout, ConnectionError) as e:
            # 网络层超时/连接错误
            wait = 0.5 * (2 ** attempt)
            if attempt < max_retries - 1:
                time.sleep(wait)
                continue
            raise Exception(f"Max retries ({max_retries}) exceeded for network error: {e}")
        except HTTPError as e:
            if e.response.status_code in [400, 429, 500, 502, 503, 504] and attempt < max_retries - 1:
                retry_after = int(e.response.headers.get("Retry-After", 0))
                wait = max(retry_after, 0.5 * (2 ** attempt))
                time.sleep(wait)
                continue
            raise Exception(f"Non-retryable HTTP error {e.response.status_code}: {e.response.text}")
    raise Exception(f"Max retries ({max_retries}) exceeded for API call")


def format_segments_for_prompt(segments: List[Dict]) -> str:
    """
    将 segments 格式化为 prompt 文本。
    """
    if not segments:
        return "无片段信息"
    
    def seg_line(s):
        return f"<START>{s.get('start')}</START> <END>{s.get('end')}</END> <SPEAKER>{s.get('speaker','UNK')}</SPEAKER>"

    lines = []
    for s in segments:
        lines.append(seg_line(s))

    header = "下列为RTTM文件提炼信息。"
    return header + "\n" + "\n".join(lines)
    
# ------------------------------
# 模型调用函数：audio/video 分别调用
# ------------------------------
def analyze_audio_worker(lang: str, audio_data_url: str, asr_text: str, segments: List[Dict], 
                         api_key: str, provider: str, model_name: str,
                          thinking_budget: int) -> Tuple[Dict, int, float]:
    segment_prompt = format_segments_for_prompt(segments)
    if lang == 'zh':
        messages = [{"role": "user", "content": [
                                {"text": AUDIO_PROMPT_TEMPLATE_zh, "type": "text"},
                                {"text": asr_text, "type": "text"},
                                {"text": segment_prompt, "type": "text"},
                                {"audio_url": {"url": audio_data_url}, "type": "audio_url"}]
                    }]
    elif lang == 'en':
        messages = [{"role": "user", "content": [
                                {"text": AUDIO_PROMPT_TEMPLATE_en, "type": "text"},
                                {"text": asr_text, "type": "text"},
                                {"text": segment_prompt, "type": "text"},
                                {"audio_url": {"url": audio_data_url}, "type": "audio_url"}]
                    }]
    resp_json = call_dashscope_api(api_key, provider, model_name, messages, thinking_budget=thinking_budget)
    # 安全检查并解析
    content_string = resp_json['choices'][0]['message']['content']
    answer_match = re.search(r"<answer>(.*?)</answer>", content_string, re.S)
    usage = resp_json.get('usage', {})
    tokens, cost = calculate_cost_from_usage(usage)
    finish_reason = resp_json['choices'][0]['finish_reason']
    if finish_reason == "length":
        print(f"[WARNING] 回复提前截断。{content_string}")
        answer_json = None
        return answer_json, tokens, cost
    if not answer_match:
        print(f"[WARNING] 未找到 <answer> 标签，响应内容不符合预期。{content_string}")
        answer_json = None
        return answer_json, tokens, cost
    answer_text = answer_match.group(1).strip()
    answer_json = json.loads(answer_text)
    return answer_json, tokens, cost


def analyze_video_worker(video_data_url: str, asr_text: str, segments: List[Dict], 
                         api_key: str, provider: str, model_name: str,
                          thinking_budget: int) -> Tuple[Dict, int, float]:
    segment_prompt = format_segments_for_prompt(segments)
    messages = [{"role": "user", "content": [
                            {"text": VIDEO_PROMPT_TEMPLATE, "type": "text"},
                            {"text": asr_text, "type": "text"},
                            {"text": segment_prompt, "type": "text"},
                            {"video_url": {"url": video_data_url}, "type": "video_url"}]
                }]
    resp_json = call_dashscope_api(api_key, provider, model_name, messages, thinking_budget=thinking_budget)
    content_string = resp_json['choices'][0]['message']['content']
    answer_match = re.search(r"<answer>(.*?)</answer>", content_string, re.S)
    usage = resp_json.get('usage', {})
    tokens, cost = calculate_cost_from_usage(usage)
    finish_reason = resp_json['choices'][0]['finish_reason']
    if finish_reason == "length":
        print(f"[WARNING] 回复提前截断。{content_string}")
        answer_json = None
        return answer_json, tokens, cost
    if not answer_match:
        print(f"[WARNING] 未找到 <answer> 标签，响应内容不符合预期。{content_string}")
        answer_json = None
        return answer_json, tokens, cost
    answer_text = answer_match.group(1).strip()
    answer_json = json.loads(answer_text)
    return answer_json, tokens, cost

# ----------------------------
# 目录遍历与 worker 调度逻辑
# ----------------------------

def process_single_rttm(rttm_path, lang, api_key, provider, model_name, thinking_budget, resume):
    meta = {"rttm": rttm_path, "status": "ok", "error": None, "tokens": 0, "cost": 0.0, "out_path": None}
    try:
        files = find_files_for_rttm(rttm_path)
        basename = files["basename"]
        parent_dir = files["parent_dir"]
        cot_wav_dir = os.path.join(parent_dir, "cot_wav")
        os.makedirs(cot_wav_dir, exist_ok=True)
        out_path = os.path.join(cot_wav_dir, f"{basename}.json")

        if resume and os.path.exists(out_path) and os.path.getsize(out_path) > 5:
            meta["status"] = "skip"
            meta["out_path"] = out_path
            meta["error"] = f"already exists"
            return meta

        if not files["wav"]:
            meta["status"] = "skip"
            meta["error"] = f"missing wav"
            return meta
        if not files["srt"]:
            meta["status"] = "skip"
            meta["error"] = f"missing srt"
            return meta

        asr_text = parse_srt(files["srt"])
        if not asr_text:
            raise ValueError(f"无法从 srt 中提取文本: {files['srt']}")
        segments = parse_rttm(rttm_path)
        
        # 1) audio 分析
        audio_data_url = read_file_as_data_url(files["wav"], "audio/wav")
        answer_json, tokens, cost = analyze_audio_worker(lang, audio_data_url, asr_text, segments,
                                                api_key, provider, model_name, thinking_budget)
        # 保存
        if answer_json:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(answer_json, f, ensure_ascii=False, indent=2)
            # 二次验证json格式
            with open(out_path, "r", encoding="utf-8") as check_f:
                obj = json.load(check_f)  # 触发解码验证

        meta["out_path"] = out_path
        meta["answer"] = answer_json
        meta["tokens"] = tokens
        meta["cost"] = cost
        return meta
    except Exception as e:
        meta["status"] = "error"
        meta["error"] = f"{repr(e)}\n{traceback.format_exc()}"
        if os.path.exists(out_path):
            os.remove(out_path)
        return meta

def batch_process(root_dir: str, lang: str, api_key: str, provider: str, model_name: str,
                  workers: int, thinking_budget: int, resume: bool):
    rttm_list = find_all_rttm_files(root_dir)
    if not rttm_list:
        print(f"[WARNING] 未找到任何 rttm 文件: {root_dir}")
        return

    print(f"[INFO] 找到 {len(rttm_list)} 个 rttm 文件，语言 {lang}，准备并发 workers={workers}")

    stats = {"total": len(rttm_list), "done": 0, "skipped": 0, "errors": 0, "tokens": 0, "cost": 0.0}
    results = []

    with ThreadPoolExecutor(max_workers=workers) as exe:
        futs = {}
        for rttm_path in rttm_list:
            fut = exe.submit(process_single_rttm, rttm_path, lang, api_key, provider, model_name, thinking_budget, resume)
            futs[fut] = rttm_path
            
        try:
            for fut in as_completed(futs):
                rttm_path = futs[fut]
                meta = fut.result()
                results.append(meta)
                if meta["status"] == "ok":
                    stats["done"] += 1
                    stats["tokens"] += meta.get("tokens", 0)
                    stats["cost"] += float(meta.get("cost", 0.0))
                    print(f"[DONE] {meta['out_path']} tokens={meta.get('tokens')} cost={meta.get('cost')} total_cost={stats['cost']} answer={meta.get('answer')}")
                elif meta["status"] == "skip":
                    stats["skipped"] += 1
                    print(f"[SKIP] {rttm_path} reason={meta.get('error')}")
                else:
                    stats["errors"] += 1
                    print(f"[ERROR] {rttm_path} err={meta.get('error')}")
        except Exception as e:
            stats["errors"] += 1
            print(f"[ERROR] {e}")

    print("===== 任务汇总 =====")
    print(f"Total: {stats['total']}, Done: {stats['done']}, Skipped: {stats['skipped']}, Errors: {stats['errors']}")
    print(f"Tokens total: {stats['tokens']}, Cost total (美元): {stats['cost']:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Multimodal emotion analysis with CoT (audio+video).")
    parser.add_argument("--root_dir", required=True, help="根目录")
    parser.add_argument("--lang", required=True, help="语言类别")
    parser.add_argument("--api_key", required=True, help="DASHSCOPE API key")
    parser.add_argument("--provider", default="google", choices=["google", "azure", "yingmao"], help="provider")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="模型名")
    parser.add_argument("--workers", type=int, default=12, help="并发 worker 数，过多容易触发限流访问被拒绝")
    parser.add_argument("--thinking_budget", type=int, default=1024, help="CoT tokens 最大值")
    parser.add_argument("--resume", action="store_true", help="结果已存在则跳过，确保断点续跑，防止重复分析")
    args = parser.parse_args()
    
    if not os.path.isdir(args.root_dir):
        print("[WARNING] root_dir 不存在或不是目录:", args.root_dir)
        sys.exit(2)
    
    batch_process(args.root_dir, args.lang, args.api_key, args.provider, args.model,
                  workers=args.workers, thinking_budget=args.thinking_budget, resume=args.resume)

if __name__ == "__main__":
    main()