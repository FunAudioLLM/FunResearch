import json
import argparse
import os
import random
import re
import time
import requests

def clean_raw_json(raw: str):
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[len("```json"):].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    return raw

def parse_json_safely(raw: str):
    cleaned = clean_raw_json(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        items = re.findall(r'\{(?:[^{}]|(?R))*\}', cleaned, re.DOTALL)
        if len(items) >= 1:
            try:
                as_arr = "[" + ",\n".join(items) + "]"
                return json.loads(as_arr)
            except Exception:
                pass
        raise

def llm_chat_json(user_query: str,
                  model_name: str = "gemini-2.5-pro",
                  system_prompt: str = None,
                  max_retries: int = 6,
                  retry_delay: int = 8,
                  timeout: int = 60):
    def remove_ctrl(s): return re.sub(r"[\x00-\x1f\x7f]", "", s)
    def is_bad(txt):
        if not txt.strip():
            return True, "Empty"
        if not (txt.lstrip().startswith("{") or txt.lstrip().startswith("[")):
            return True, f"Non-JSON: {txt[:100]}"
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict) and "error" in obj:
                return True, f"API error: {obj.get('error')}"
        except json.JSONDecodeError:
            return True, f"Invalid JSON: {txt[:100]}"
        return False, None

    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    api_key = os.environ.get("DASHSCOPE_API_KEY", "YOUR_API_KEY")
    headers = {"content-type": "application/json", "Authorization": api_key}

    messages = []
    if (system_prompt or "").strip():
        messages.append({"role": "system", "content": system_prompt})
    uq = user_query.strip() or "OK"
    messages.append({"role": "user", "content": uq})

    payload = {
        "model": model_name,
        "messages": messages,
        "n": 1,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            txt = remove_ctrl(resp.text)
            bad, reason = is_bad(txt)
            if bad:
                raise RuntimeError(reason)
            obj = json.loads(txt)
            content = obj["choices"][0]["message"]["content"]
            return content
        except Exception:
            if attempt < max_retries:
                time.sleep(retry_delay * attempt + random.uniform(0, 1.0))
                continue
            raise

def get_examples(data):
    if isinstance(data, dict) and "examples" in data:
        return data["examples"]
    if isinstance(data, list):
        return data
    raise ValueError("Input JSON must be a list or contain an 'examples' list.")

def has_multi_calls(examples):
    for e in examples:
        if "function_calls" in e and isinstance(e["function_calls"], list) and len(e["function_calls"]) > 1:
            return True
    return False

def normalize_for_single(examples):
    normalized = []
    for entry in examples:
        if "function_calls" in entry and isinstance(entry["function_calls"], list) and len(entry["function_calls"]) == 1:
            call = entry["function_calls"][0]
            normalized.append({
                "question": entry.get("question", ""),
                "function": (call.get("function") or "").strip(),
                "description": call.get("description", ""),
                "parameters": call.get("parameters", {}) or {}
            })
        elif "function" in entry and "parameters" in entry:
            normalized.append({
                "question": entry.get("question", ""),
                "function": (entry.get("function") or "").strip(),
                "description": entry.get("description", ""),
                "parameters": entry.get("parameters", {}) or {}
            })
    return normalized

def load_name_to_tool_def(tool_path):
    with open(tool_path, "r", encoding="utf-8") as f:
        pool = json.load(f)
    name_to_def = {}
    if isinstance(pool, list):
        for item in pool:
            if isinstance(item, dict) and "function" in item and isinstance(item["function"], list):
                for func in item["function"]:
                    if not isinstance(func, dict):
                        continue
                    nm = (func.get("name") or "").strip()
                    if nm:
                        name_to_def[nm] = func
            elif isinstance(item, dict) and "name" in item:
                nm = (item.get("name") or "").strip()
                if nm:
                    name_to_def[nm] = item
    else:
        raise ValueError("Tool pool JSON must be a list")
    return name_to_def

def _tool_to_json_spec(tool_def: dict) -> dict:
    name = (tool_def.get("name") or "").strip()
    desc = tool_def.get("description", "")
    params = tool_def.get("parameters", {}) or {}
    if "type" not in params:
        params["type"] = "object"
    if "properties" not in params:
        params["properties"] = {}
    if "required" not in params:
        params["required"] = []
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": params
        }
    }

def _join_tools_as_objects_lines(tool_defs) -> str:
    return "\n".join(json.dumps(_tool_to_json_spec(td), ensure_ascii=False) for td in tool_defs)

# === Sample Agent ===
def format_tools_pool_for_prompt(name_to_def: dict) -> str:
    lines = []
    for name, tdef in name_to_def.items():
        lines.append(json.dumps({"name": name, "description": tdef.get("description", "")}, ensure_ascii=False))
    return "\n".join(lines)

def build_sample_agent_system_prompt(tools_pool_text: str) -> str:
    return (
        "You are an expert in tool composition. You need to select one or more functionally related tools (up to five) from the tool pool and combine them into a toolkit that has the potential to solve a specific user problem.\n\n"
        f"Tools Pool: {tools_pool_text}\n\n"
        "In addition to the tool above you have selected, you are also required to choose several tools from the tools pool as distractors: some that are highly related to your chosen combination, and some that are less related. Finally, return all selected tools (including the combination and distractors) in the specified JSON format: {tools}"
    )

def sample_agent_select_distractors(focus_functions, name_to_def: dict, model_name: str, want_total: int = 4) -> list:
    if isinstance(focus_functions, str):
        focus_list = [focus_functions.strip()]
    else:
        focus_list = [str(x).strip() for x in focus_functions if str(x).strip()]
    tools_pool_text = format_tools_pool_for_prompt(name_to_def)
    system_prompt = build_sample_agent_system_prompt(tools_pool_text)
    user_query = (
        "Please focus on a combination that includes: "
        + ", ".join(focus_list)
        + ". Return JSON strictly (no extra text): "
        + '{"combination":["tool_name",...], "distractors":{"highly_related":["..."], "less_related":["..."]}}'
    )
    less, rel = [], []
    try:
        content = llm_chat_json(user_query, model_name=model_name, system_prompt=system_prompt)
        parsed = parse_json_safely(content)
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        if isinstance(parsed, dict):
            dist = parsed.get("distractors", {}) if isinstance(parsed.get("distractors"), dict) else {}
            less = dist.get("less_related", []) if isinstance(dist.get("less_related"), list) else []
            rel = dist.get("highly_related", []) if isinstance(dist.get("highly_related"), list) else []
    except Exception:
        pass
    picked = []
    for n in less + rel:
        n = (n or "").strip()
        if n and n not in focus_list and n in name_to_def and n not in picked:
            picked.append(n)
        if len(picked) >= want_total:
            break
    if len(picked) < want_total:
        candidates = [nm for nm in name_to_def.keys() if nm not in focus_list and nm not in picked]
        random.shuffle(candidates)
        for nm in candidates:
            picked.append(nm)
            if len(picked) >= want_total:
                break
    return [name_to_def[nm] for nm in picked if nm in name_to_def]

def convert_single_with_agent(examples, name_to_def, num_distractors: int, model_name: str):
    result = []
    for entry in examples:
        question_text = entry.get("question", "")
        function_name = (entry.get("function") or "").strip()
        parameters = entry.get("parameters", {}) or {}
        main_def = name_to_def.get(function_name) or {"name": function_name, "description": entry.get("description", ""), "parameters": {"type": "object", "properties": {}, "required": []}}
        distractor_defs = sample_agent_select_distractors(function_name, name_to_def, model_name=model_name, want_total=num_distractors)
        all_tools = [main_def] + distractor_defs
        seen, uniq = set(), []
        for t in all_tools:
            nm = (t.get("name") or "").strip()
            if nm and nm not in seen:
                uniq.append(t)
                seen.add(nm)
        tools_block = _join_tools_as_objects_lines(uniq)
        instruction = (
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            f"{tools_block}\n"
            "</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )
        new_instruction = f"{instruction}\n\nUser:\n{question_text}"
        output = "<tool_call>\n" + json.dumps({"name": function_name, "arguments": parameters}, ensure_ascii=False) + "\n</tool_call>"
        result.append({"instruction": new_instruction, "input": "", "output": output})
    return result

def convert_multi_with_agent(examples, name_to_def, num_distractors: int, model_name: str):
    result = []
    for entry in examples:
        if "function_calls" not in entry or not isinstance(entry["function_calls"], list) or not entry["function_calls"]:
            continue
        question_text = entry.get("question", "")
        used_names, main_defs = [], []
        for call in entry["function_calls"]:
            fname = (call.get("function") or "").strip()
            if not fname:
                continue
            if fname not in used_names:
                used_names.append(fname)
                defn = name_to_def.get(fname) or {"name": fname, "description": call.get("description", ""), "parameters": {"type": "object", "properties": {}, "required": []}}
                main_defs.append(defn)
        distractor_defs = sample_agent_select_distractors(used_names, name_to_def, model_name=model_name, want_total=num_distractors)
        all_tools = main_defs + [d for d in distractor_defs if (d.get("name","").strip() not in {m.get("name","").strip() for m in main_defs})]
        seen, uniq = set(), []
        for t in all_tools:
            nm = (t.get("name") or "").strip()
            if nm and nm not in seen:
                uniq.append(t)
                seen.add(nm)
        tools_block = _join_tools_as_objects_lines(uniq)
        instruction = (
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            f"{tools_block}\n"
            "</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )
        new_instruction = f"{instruction}\n\nUser question:\n{question_text}"
        blocks = []
        for call in entry["function_calls"]:
            payload = {"name": call["function"], "arguments": call["parameters"]}
            blocks.append("<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>")
        output = "\n".join(blocks)
        result.append({"instruction": new_instruction, "input": "", "output": output})
    return result

def main():
    parser = argparse.ArgumentParser(description="Convert function-call data (single or multi-tool) to instruction-tuning format with Sample Agent distractor sampling.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--tool_path", required=True)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--num_distractors", type=int, default=4)
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"❌ 输入文件不存在: {args.input_path}")
        return
    if not os.path.exists(args.tool_path):
        print(f"❌ 工具池文件不存在: {args.tool_path}")
        return

    with open(args.input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    examples = get_examples(raw)
    name_to_def = load_name_to_tool_def(args.tool_path)

    if has_multi_calls(examples):
        instruct_data = convert_multi_with_agent(examples, name_to_def, num_distractors=args.num_distractors, model_name=args.model)
    else:
        singles = normalize_for_single(examples)
        instruct_data = convert_single_with_agent(singles, name_to_def, num_distractors=args.num_distractors, model_name=args.model)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(instruct_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 转换完成，共处理 {len(instruct_data)} 条数据，输出至：{args.output_path}")

if __name__ == "__main__":
    main()
