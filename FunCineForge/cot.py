#!/usr/bin/env python3
"""
用音视频大模型做 audio / video CoT分析。
我们依赖: requests url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    本工具推荐并采用 gemini-2.5-flash model， 可自行更换其他多模态大模型
    python cot.py --root_dir datasets/clean/zh --provider google --model gemini-2.5-flash --api_key xxx --resume
可更换url依赖，参考https://ai.google.dev/gemini-api/docs/structured-output，需修改call_dashscope_api
from google import genai
client = genai.Client()
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=messages,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=2048)),
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import requests
import time
import gc

# ------------------------------
# 构造 Prompt（音频 / 视频）
# ------------------------------
AUDIO_PROMPT_TEMPLATE = """
你是一个专注于音频分析的专家，分析的对象是给定的wav音频，辅助信息有ASR转录文本，RTTM中提取的时间戳和对应的说话人。
注意ASR转录文本存在词汇错误和标点断句错误的情况，RTTM中的speaker id可能少标或多标,请根据实际音色区分id，但正确的id需要和RTTM中的说话人id匹配。
任务：判断有几个说话人，判断每个说话人的年龄性别和音色，
准确识别语音并基于提供的ASR转录文本，文本纠错后得到正确的文本和合理的标点。
根据语音内容和RTTM辅助信息，理解音频的各个时间段是哪个说话人在说话，
然后思考并总结音频的整体情感线索和每个说话人的韵律情感变化，总结每个说话人的信息和情感线索并保存在clue中，线索用中文陈述，不用描述背景音效，线索限制在85字以内。
输出：
严格按照下面的格式输出，无其他任何额外输出。
label为<中性、不确定、喜悦、信任、害怕、惊讶、难过、厌恶、生气、期待、紧张>中一种，confidence 为label的置信度，text为纠正的转录文本，
id 匹配RTTM中说话人1,2,3...，age为<儿童，青年，中年，中老年，老年，不确定>中一种，gender为<男，女，不确定>中一种，timbre为几个描述音色的词汇。
<answer>{"label": "中性", "confidence": 0.8, "text": "哎呀，将军，将军，不可连累老夫啊！大丈夫生居天地之间，岂能郁郁久居人下！", "speakers": [{"id": "1", "age": "中年", "gender": "男", "timbre": "低沉、苍老"},{"id": "2", "age": "青年", "gender": "男", "timbre": "高亢、有力、果断"},...], "clue": "两名角色对话，第一位中年男性角色情绪紧张，略带颤抖和哀求，表达对被牵连的恐惧。第二位角色语调变得激昂坚定，铿锵有力，充满对尊严和自由的强烈渴望。整体展现出从畏惧到反抗的情感转变。"}</answer>
"""

VIDEO_PROMPT_TEMPLATE = """
"""


# ------------------------------
# 辅助函数
# ------------------------------
# 价格（美元 / 每百万 token）
_PRICE_TEXT_PER_M = 0.4 / 1_000_000
_PRICE_AUDIO_PER_M = 0.7 / 1_000_000
_PRICE_OUTPUT_PER_M = 0.4 / 1_000_000

def calculate_cost_from_usage(usage: Dict[str, Any]) -> Tuple[int, float]:
    """
    根据 usage 字段计算成本, 对缺失字段使用 0 作为默认值。
    """
    prompt_details = usage.get('prompt_tokens_details', {}) or {}
    audio_tokens = int(prompt_details.get('audio_tokens', 0) or 0)
    text_tokens = int(prompt_details.get('text_tokens', 0) or 0)

    completion_tokens = int(usage.get('completion_tokens', 0) or 0)
    think_tokens = int(usage.get('think_tokens', 0) or 0)
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
                       thinking_budget: int = 2048, min_tokens: int = 1024, timeout: int = 120) -> Dict:
    """
    调用 DashScope API，失败时自动重试一次
    """
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": api_key}

    last_err = None
    for attempt in range(5):
        try:
            if attempt == 4:
                thinking_budget = 1024
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": min_tokens * (6 - attempt),
                "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
                "dashscope_extend_params": {"provider": provider}
            }
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < 4:
                time.sleep(1)  # 重试前等 1 秒
            else:
                raise last_err

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
def analyze_audio_worker(audio_data_url: str, asr_text: str, segments: List[Dict], 
                         api_key: str, provider: str, model_name: str,
                          thinking_budget: int) -> Tuple[Dict, int, float]:
    segment_prompt = format_segments_for_prompt(segments)
    messages = [{"role": "user", "content": [
                            {"text": AUDIO_PROMPT_TEMPLATE, "type": "text"},
                            {"text": asr_text, "type": "text"},
                            {"text": segment_prompt, "type": "text"},
                            {"audio_url": {"url": audio_data_url}, "type": "audio_url"}]
                }]
    resp_json = call_dashscope_api(api_key, provider, model_name, messages, thinking_budget=thinking_budget)
    # 安全检查并解析
    content_string = resp_json['choices'][0]['message']['content']
    usage = resp_json.get('usage', {})
    tokens, cost = calculate_cost_from_usage(usage)
    answer_match = re.search(r"<answer>(.*?)</answer>", content_string, re.S)
    if not answer_match:
        print("[WARNING] 未找到 <answer> 标签，响应内容不符合预期。")
        answer_json = {}
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
                            {"video_url": {"url": video_data_url}, "type": "video_url"},
                            {"text": asr_text, "type": "text"},
                            {"text": segment_prompt, "type": "text"}]
                }]
    resp_json = call_dashscope_api(api_key, provider, model_name, messages, thinking_budget=thinking_budget)
    content_string = resp_json['choices'][0]['message']['content']
    answer_match = re.search(r"<answer>(.*?)</answer>", content_string, re.S)
    answer_text = answer_match.group(1).strip()
    answer_json = json.loads(answer_text)
    return answer_json

# ----------------------------
# 目录遍历与 worker 调度逻辑
# ----------------------------

def process_single_rttm(rttm_path, api_key, provider, model_name, thinking_budget, resume):
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
        answer_json, tokens, cost = analyze_audio_worker(audio_data_url, asr_text, segments,
                                                api_key, provider, model_name, thinking_budget)
        # 保存
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(answer_json, f, ensure_ascii=False, indent=2)


        meta["out_path"] = out_path
        meta["tokens"] = tokens
        meta["cost"] = cost
        return meta
    except Exception as e:
        meta["status"] = "error"
        meta["error"] = f"{repr(e)}\n{traceback.format_exc()}"
        return meta

def batch_process(root_dir: str, api_key: str, provider: str, model_name: str,
                  workers: int, thinking_budget: int, resume: bool):
    rttm_list = find_all_rttm_files(root_dir)
    if not rttm_list:
        print(f"[WARNING] 未找到任何 rttm 文件: {root_dir}")
        return

    print(f"[INFO] 找到 {len(rttm_list)} 个 rttm 文件，准备并发 workers={workers}")

    stats = {"total": len(rttm_list), "done": 0, "skipped": 0, "errors": 0, "tokens": 0, "cost": 0.0}
    results = []

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futs = {}
        for rttm_path in rttm_list:
            fut = exe.submit(process_single_rttm, rttm_path, api_key, provider, model_name, thinking_budget, resume)
            futs[fut] = rttm_path

        for fut in as_completed(futs):
            rttm_path = futs[fut]
            try:
                meta = fut.result()
                results.append(meta)
                if meta["status"] == "ok":
                    stats["done"] += 1
                    stats["tokens"] += meta.get("tokens", 0)
                    stats["cost"] += float(meta.get("cost", 0.0))
                    print(f"[DONE] {meta['out_path']} tokens={meta.get('tokens')} cost={meta.get('cost')} total_cost={stats['cost']}")
                elif meta["status"] == "skip":
                    stats["skipped"] += 1
                    print(f"[SKIP] {rttm_path} reason={meta.get('error')}")
                else:
                    stats["errors"] += 1
                    print(f"[ERROR] {rttm_path} err={meta.get('error')}")
            except Exception as e:
                stats["errors"] += 1
                print(f"[EXCEPTION] 在处理 {rttm_path} 时失败: {repr(e)}")

    print("===== 任务汇总 =====")
    print(f"Total: {stats['total']}, Done: {stats['done']}, Skipped: {stats['skipped']}, Errors: {stats['errors']}")
    print(f"Tokens total: {stats['tokens']}, Cost total (美元): {stats['cost']:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Multimodal emotion analysis with CoT (audio+video).")
    parser.add_argument("--root_dir", required=True, help="根目录")
    parser.add_argument("--api_key", required=True, help="DASHSCOPE API key")
    parser.add_argument("--provider", default="google", choices=["google", "azure", "yingmao"], help="provider")
    parser.add_argument("--model", default="gemini-2.5-flash", help="模型名")
    parser.add_argument("--workers", type=int, default=12, help="并发 worker 数")
    parser.add_argument("--thinking_budget", type=int, default=2048, help="CoT tokens 预算")
    parser.add_argument("--resume", action="store_true", help="结果已存在则跳过，确保断点续跑，防止重复分析")
    args = parser.parse_args()
    
    if not os.path.isdir(args.root_dir):
        print("[WARNING] root_dir 不存在或不是目录:", args.root_dir)
        sys.exit(2)
    
    batch_process(args.root_dir, args.api_key, args.provider, args.model,
                  workers=args.workers, thinking_budget=args.thinking_budget, resume=args.resume)

if __name__ == "__main__":
    main()