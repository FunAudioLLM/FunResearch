import re
import random
import json
import requests
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate tool-use training data with multi-agent pipeline and pairwise selection.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--num_questions", type=int, default=5)
    parser.add_argument("--mode", type=str, default="mode1", choices=["mode1", "mode2", "mode3", "mode4"])
    return parser.parse_args()

def clean_raw_json(raw):
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[len("```json"):].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    return raw

def single_search(query, model_name="gemini-2.5-pro", n=1, temperature=1.0,
                  do_print=False, json_mode=True, max_retries=8, retry_delay=10, timeout=60,
                  system_prompt=None):
    def remove_control_characters(s):
        return re.sub(r"[\x00-\x1f\x7f]", "", s)

    def is_transient_error_or_invalid(response_text):
        if not response_text.strip():
            return True, "Empty response text"
        try:
            obj = json.loads(response_text)
            if isinstance(obj, dict) and "error" in obj:
                return True, f"API error: {json.dumps(obj['error'], ensure_ascii=False)}"
        except json.JSONDecodeError:
            return True, f"Invalid JSON envelope: {response_text[:120]}"
        return False, None

    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    GPT_AUTHORIZATION = "YOUR_API_KEY"
    headers = {"content-type": "application/json", "Authorization": f"{GPT_AUTHORIZATION}"}

    if query is None:
        query = ""
    if not isinstance(query, str):
        query = str(query)
    if not query.strip():
        if system_prompt and isinstance(system_prompt, str) and system_prompt.strip():
            query = system_prompt
            system_prompt = None
        else:
            query = "OK"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    payload = {"model": model_name, "messages": messages, "n": n, "temperature": temperature}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            clean_text = remove_control_characters(response.text)
            need_retry, reason = is_transient_error_or_invalid(clean_text)
            if need_retry:
                raise RuntimeError(reason)
            if do_print:
                print(f"> Clean response: {clean_text[:200]}")
            time.sleep(random.uniform(2.0, 4.0))
            return clean_text
        except Exception as e:
            print(f"⚠️ Exception on attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                delay = retry_delay * attempt + random.uniform(0, 1.0)
                print(f"🔁 Retrying in {delay:.1f} seconds...\n")
                time.sleep(delay)
                continue
            else:
                print("❌ Max retries reached. Skipping.\n")
                break
    raise RuntimeError(f"Failed to get a valid response after {max_retries} retries.")

def parse_search_result(search_result):
    search_result_item = json.loads(search_result)
    if "error" in search_result_item:
        raise RuntimeError(f"API error: {json.dumps(search_result_item['error'], ensure_ascii=False)}")
    try:
        response = search_result_item["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Invalid model response structure: {search_result_item}") from e
    if not isinstance(response, str) or not response.strip():
        raise RuntimeError(f"Empty content in model response: {search_result_item}")
    return {"response": response}

def is_list_of_tools(function_field):
    return isinstance(function_field, list)

def extract_schema_params(function_def):
    params = function_def.get("parameters", {}) if isinstance(function_def.get("parameters"), dict) else {}
    required = params.get("required", []) if isinstance(params.get("required"), list) else []
    properties = params.get("properties", {}) if isinstance(params.get("properties"), dict) else {}
    all_keys = list(properties.keys())
    optional = [k for k in all_keys if k not in required]
    return required, optional

def coerce_type(value, schema):
    t = (schema.get("type") or "").lower()
    if t in ("integer", "int"):
        if isinstance(value, str):
            v = value.strip()
            if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                return int(v)
        if isinstance(value, (float, int)):
            return int(value)
    elif t in ("number", "float", "double"):
        if isinstance(value, str):
            v = value.strip()
            try:
                return float(v)
            except Exception:
                return value
        if isinstance(value, (int, float)):
            return float(value)
    elif t in ("boolean", "bool"):
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ["true", "yes", "1", "on"]:
                return True
            if v in ["false", "no", "0", "off"]:
                return False
    elif t in ("string", "str"):
        return str(value) if not isinstance(value, str) else value
    return value

def validate_and_coerce(function_def, params):
    errors = []
    coerced = dict(params or {})
    parameters = function_def.get("parameters", {}) if isinstance(function_def.get("parameters"), dict) else {}
    props = parameters.get("properties", {}) if isinstance(parameters.get("properties"), dict) else {}
    required, _ = extract_schema_params(function_def)
    for k in required:
        if k not in coerced:
            errors.append(f"Missing required param: {k}")
    for k, v in list(coerced.items()):
        if k in props and isinstance(props[k], dict):
            coerced[k] = coerce_type(v, props[k])
    return (len(errors) == 0), coerced, errors

# ===================== Memory Agent =====================

def memory_agent_summarize(history_items, model_name="gemini-2.5-pro"):
    history_text = json.dumps(history_items, ensure_ascii=False, indent=2)
    system_prompt = (
        "You are a memory-aware agent. You need to store all previous historical dialogue records provided to you and classify each of them into an appropriate category based on its content and intent.\n\n"
        f"History Dialogue: {history_text}\n\n"
        "Now, you need to share all historical dialogue records along with your categorized judgments with the dialogue generation system, and provide corresponding guidance to help the system generate more diverse user queries.\n"
        "Return specified JSON format:\n"
        "{\n"
        '  "summary": "brief summary",\n'
        '  "categories": [{"label":"...", "count": N}],\n'
        '  "guidance": "actionable guidance to diversify"\n'
        "}"
    )
    raw = single_search("", model_name=model_name, temperature=0.0, json_mode=True, system_prompt=system_prompt)
    content = parse_search_result(raw)
    obj = json.loads(clean_raw_json(content["response"]))
    return obj

def memory_agent_classify(question, model_name="gemini-2.5-pro"):
    system_prompt = (
        "You are a memory-aware agent. Classify the question by its type/topic/intent.\n"
        f'Question: "{question}"\n'
        'Return specified JSON format: {"category":"..."}'
    )
    raw = single_search("", model_name=model_name, temperature=0.0, json_mode=True, system_prompt=system_prompt)
    content = parse_search_result(raw)
    obj = json.loads(clean_raw_json(content["response"]))
    return obj.get("category", "unknown")

# ===================== Function Agent =====================

def build_function_agent_prompt(question, tool_list, chosen_optional_map):
    is_multi = len(tool_list) > 1
    if is_multi:
        func_json = json.dumps(tool_list, ensure_ascii=False, indent=2)
        required_struct = []
        chosen_struct = []
        for i, tool_def in enumerate(tool_list):
            required, _ = extract_schema_params(tool_def)
            allowed_opt = chosen_optional_map[i] if i < len(chosen_optional_map) else []
            required_struct.append({
                "index": i,
                "function": tool_def.get("name"),
                "required": required
            })
            chosen_struct.append({
                "index": i,
                "function": tool_def.get("name"),
                "allowed_optional": allowed_opt
            })
        required_text = json.dumps(required_struct, ensure_ascii=False, indent=2)
        chosen_text = json.dumps(chosen_struct, ensure_ascii=False, indent=2)
        tools_format = (
            "[\n"
            '  {"function":"name","description":"desc","parameters":{...}},\n'
            "  ...\n"
            "]"
        )
    else:
        tool_def = tool_list[0]
        func_json = json.dumps(tool_def, ensure_ascii=False, indent=2)
        required, _ = extract_schema_params(tool_def)
        allowed_opt = chosen_optional_map[0] if chosen_optional_map else []
        required_text = json.dumps(required, ensure_ascii=False)
        chosen_text = json.dumps(allowed_opt, ensure_ascii=False)
        tools_format = (
            "{\n"
            '  "function": "name",\n'
            '  "description": "desc",\n'
            '  "parameters": { ... }\n'
            "}"
        )
    return (
        "You are an expert in selecting tool functions and determining their relevant parameters.\n\n"
        f"User query: {question}\n"
        f"Function definition: {func_json}\n"
        f"Required parameters: {required_text}\n"
        f"Optional parameters: {chosen_text}\n\n"
        "When extracting parameters from the user's query, accurately identify the values for required parameters. "
        "For optional parameters, flexibly decide how many to include, ensuring diversity in parameter selection. "
        "Finally, return the tool call in the specified JSON format: \n"
        f"{tools_format}"
    )

def pick_optional_by_percentage(optional_keys):
    if not optional_keys:
        return set()
    pct = random.random()
    k = int(round(len(optional_keys) * pct))
    k = max(0, min(k, len(optional_keys)))
    if k == 0:
        return set()
    return set(random.sample(optional_keys, k))

def function_agent_extract(question, func_field, model_name="gemini-2.5-pro"):
    if isinstance(func_field, dict):
        tool_list = [func_field]
    elif isinstance(func_field, list):
        tool_list = func_field
    else:
        raise ValueError("func_field must be a tool object or a list of tool objects.")
    chosen_optional_map = []
    for tool_def in tool_list:
        _, optional = extract_schema_params(tool_def)
        chosen_optional_map.append(list(pick_optional_by_percentage(optional)))
    prompt = build_function_agent_prompt(question, tool_list, chosen_optional_map)
    raw = single_search(prompt, model_name=model_name, temperature=0.0, json_mode=True)
    content = parse_search_result(raw)
    parsed = json.loads(clean_raw_json(content["response"]))
    is_multi = len(tool_list) > 1
    if is_multi:
        if not isinstance(parsed, list):
            return []
        calls_in = parsed
    else:
        if not isinstance(parsed, dict):
            return []
        calls_in = [parsed]
    results = []
    for idx, tool_def in enumerate(tool_list):
        item = calls_in[idx] if idx < len(calls_in) else {}
        params = item.get("parameters", {}) if isinstance(item.get("parameters"), dict) else {}
        required, optional = extract_schema_params(tool_def)
        allowed = set(chosen_optional_map[idx]) if idx < len(chosen_optional_map) else set()
        filtered = {}
        for k in required:
            if k in params:
                filtered[k] = params[k]
        for k in allowed:
            if k in params:
                filtered[k] = params[k]
        ok, coerced, _ = validate_and_coerce(tool_def, filtered)
        if not ok:
            for k in required:
                if k not in coerced and k in params:
                    coerced[k] = params[k]
            ok2, coerced, _ = validate_and_coerce(tool_def, coerced)
            if not ok2:
                continue
        results.append({
            "function": item.get("function", tool_def.get("name")),
            "description": item.get("description", tool_def.get("description", "")),
            "parameters": coerced
        })
    return results

def build_function_agent_prompt_single(question, tool_def, chosen_optional):
    return build_function_agent_prompt(
        question,
        [tool_def],
        [list(chosen_optional) if isinstance(chosen_optional, (list, set, tuple)) else []]
    )

def build_function_agent_prompt_multiple(question, tool_list, chosen_optional_map):
    return build_function_agent_prompt(question, tool_list, chosen_optional_map)

def function_agent_extract_single(question, tool_def, model_name="gemini-2.5-pro"):
    calls = function_agent_extract(question, tool_def, model_name=model_name)
    return calls[0] if calls else None

def function_agent_extract_multiple(question, tool_list, model_name="gemini-2.5-pro"):
    return function_agent_extract(question, tool_list, model_name=model_name)

# ===================== Judge Agent =====================

def judge_agent_choose(candidate_a, candidate_b, model_name="gemini-2.5-pro"):
    def candidate_calls(cand):
        if "function_calls" in cand and isinstance(cand["function_calls"], list):
            return cand["function_calls"]
        if "function" in cand:
            return [{"function": cand["function"], "description": cand.get("description", ""), "parameters": cand.get("parameters", {})}]
        return []
    def is_valid_candidate(cand):
        calls = candidate_calls(cand)
        if not calls:
            return False
        return all(isinstance(c.get("parameters"), dict) and len(c.get("parameters")) > 0 for c in calls)
    valid_a = is_valid_candidate(candidate_a)
    valid_b = is_valid_candidate(candidate_b)
    if valid_a and not valid_b:
        return {"winner": "A"}
    if valid_b and not valid_a:
        return {"winner": "B"}
    if not valid_a and not valid_b:
        return {"winner": "none"}
    system_prompt = (
        "You are an expert in evaluating sentence quality. You are evaluating two generated tool-usage problems.\n"
        "Please evaluate the provided candidate questions or responses based on two key criteria:\n"
        "(1) the practical significance of the problem they address.\n"
        "(2) the suitability of the selected tools for solving that problem.\n"
        "Choose the one that better balances real-world relevance and effective tool alignment.\n"
        'Return JSON: {"winner":"A" or "B"}'
    )
    cand_a_text = json.dumps(candidate_a, ensure_ascii=False, indent=2)
    cand_b_text = json.dumps(candidate_b, ensure_ascii=False, indent=2)
    user_query = f"Candidate A:\n{cand_a_text}\n\nCandidate B:\n{cand_b_text}\n"
    raw = single_search(user_query, model_name=model_name, temperature=0.0, json_mode=True, system_prompt=system_prompt)
    content = parse_search_result(raw)
    obj = json.loads(clean_raw_json(content["response"]))
    return {"winner": obj.get("winner", "A")}

def build_generation_prompt_mode1_single(tool_def, memory_guidance=None):
    func_json = json.dumps(tool_def, ensure_ascii=False, indent=2)
    guidance_text = ""
    if memory_guidance:
        guidance_text = (
            f"\nMemory Guidance:\n"
            f"Summary: {memory_guidance.get('summary','')}\n"
            f"Categories: {json.dumps(memory_guidance.get('categories', []), ensure_ascii=False)}\n"
            f"Guidance: {memory_guidance.get('guidance','')}\n"
        )
    return (
        "Generate 1 meaningful, realistic problem that can be solved via exactly one call to the given tool.\n"
        f"{guidance_text}\n"
        f"Tool:\n{func_json}\n"
        'Return JSON: {"question":"..."}'
    )

def build_generation_prompt_mode1_multiple(tool_list, memory_guidance=None):
    tools_json = json.dumps(tool_list, ensure_ascii=False, indent=2)
    guidance_text = ""
    if memory_guidance:
        guidance_text = (
            f"\nMemory Guidance:\n"
            f"Summary: {memory_guidance.get('summary','')}\n"
            f"Categories: {json.dumps(memory_guidance.get('categories', []), ensure_ascii=False)}\n"
            f"Guidance: {memory_guidance.get('guidance','')}\n"
        )
    return (
        "Generate 1 practical problem that requires using ALL the provided tools (one call per tool) to solve.\n"
        f"{guidance_text}\n"
        f"Tools:\n{tools_json}\n"
        'Return JSON: {"question":"..."}'
    )

def build_generation_prompt_mode2_single(tool_def, memory_guidance=None):
    func_json = json.dumps(tool_def, ensure_ascii=False, indent=2)
    guidance_text = ""
    if memory_guidance:
        guidance_text = (
            f"\nMemory Guidance:\n"
            f"Summary: {memory_guidance.get('summary','')}\n"
            f"Categories: {json.dumps(memory_guidance.get('categories', []), ensure_ascii=False)}\n"
            f"Guidance: {memory_guidance.get('guidance','')}\n"
        )
    return (
        "Generate 1 meaningful, realistic problem that can be solved by the given tool. This problem must contain several tasks that the user wants to solve. The tool must be call at least 2 times\n"
        f"{guidance_text}\n"
        f"Tool:\n{func_json}\n"
        'Return JSON: {"question":"..."}'
    )

def build_generation_prompt_mode2_multiple(tool_list, memory_guidance=None):
    tools_json = json.dumps(tool_list, ensure_ascii=False, indent=2)
    guidance_text = ""
    if memory_guidance:
        guidance_text = (
            f"\nMemory Guidance:\n"
            f"Summary: {memory_guidance.get('summary','')}\n"
            f"Categories: {json.dumps(memory_guidance.get('categories', []), ensure_ascii=False)}\n"
            f"Guidance: {memory_guidance.get('guidance','')}\n"
        )
    return (
        "Generate 1 practical problem that requires using ALL the provided tools to solve. This problem must contain several tasks that the user wants to solve. Some of the tools must be call at least 2 times\n"
        f"{guidance_text}\n"
        f"Tools:\n{tools_json}\n"
        'Return JSON: {"question":"..."}'
    )

def build_generation_prompt_mode3_single(tool_def, memory_guidance=None):
    func_json = json.dumps(tool_def, ensure_ascii=False, indent=2)
    guidance_text = ""
    if memory_guidance:
        guidance_text = (
            f"\nMemory Guidance:\n"
            f"Summary: {memory_guidance.get('summary','')}\n"
            f"Categories: {json.dumps(memory_guidance.get('categories', []), ensure_ascii=False)}\n"
            f"Guidance: {memory_guidance.get('guidance','')}\n"
        )
    return (
        "Your task is to generate 1 meaningful multi-turn dialogue that can be solved using the given tool. The tool must be called exactly once.\n"
        f"{guidance_text}\n"
        "Requirements:\n"
        "1) Write alternating lines:\n"
        "   user: ...\n"
        "   system: ...\n"
        "   user: ...\n"
        "   system: ... (3 to 5 turns)\n"
        "2) The conversation should gradually reveal all information needed for exactly one tool call.\n"
        "3) In the FINAL user turn, explicitly list all required parameter values needed by the tool (use concrete numbers and clear wording). Example: 'base=12, height=7'.\n"
        "Return JSON:\n"
        '{"question":"user: ...\\nsystem: ...\\nuser: ..."}\n'
        f"\nTool schema:\n{func_json}\n"
    )

def build_generation_prompt_mode3_multiple(tool_list, memory_guidance=None):
    tools_json = json.dumps(tool_list, ensure_ascii=False, indent=2)
    guidance_text = ""
    if memory_guidance:
        guidance_text = (
            f"\nMemory Guidance:\n"
            f"Summary: {memory_guidance.get('summary','')}\n"
            f"Categories: {json.dumps(memory_guidance.get('categories', []), ensure_ascii=False)}\n"
            f"Guidance: {memory_guidance.get('guidance','')}\n"
        )
    return (
        "Your task is to generate 1 meaningful multi-turn dialogue that must use ALL the given tools, each exactly once.\n"
        f"{guidance_text}\n"
        "Requirements:\n"
        "1) Write alternating lines:\n"
        "   user: ...\n"
        "   system: ...\n"
        "   user: ...\n"
        "   system: ... (3 to 5 turns)\n"
        "2) The conversation should gradually reveal information needed for one call per tool.\n"
        "3) In the FINAL user turn, explicitly list all required parameter values for EACH tool with concrete numbers. You may use simple labels per tool. Example:\n"
        "   'triangle: side1=3, side2=4, side3=5; circle: radius=2.5'.\n"
        "Return JSON:\n"
        '{"question":"user: ...\\nsystem: ...\\nuser: ..."}\n'
        f"\nTools schema:\n{tools_json}\n"
    )

def build_generation_prompt_mode4_single(tool_def, memory_guidance=None):
    func_json = json.dumps(tool_def, ensure_ascii=False, indent=2)
    guidance_text = ""
    if memory_guidance:
        guidance_text = (
            f"\nMemory Guidance:\n"
            f"Summary: {memory_guidance.get('summary','')}\n"
            f"Categories: {json.dumps(memory_guidance.get('categories', []), ensure_ascii=False)}\n"
            f"Guidance: {memory_guidance.get('guidance','')}\n"
        )
    return (
        "Your task is to generate 1 meaningful multi-turn dialogue solvable with the given tool. The dialogue should contain multiple user tasks, and the tool will be called at least twice.\n"
        f"{guidance_text}\n"
        "Requirements:\n"
        "1) Write alternating lines (3 to 6 turns).\n"
        "2) Ensure each intended tool call is supported by explicit parameter values.\n"
        "3) In the FINAL user turn, explicitly list all required parameter values for the NEXT immediate tool call with concrete numbers.\n"
        "Return JSON:\n"
        '{"question":"user: ...\\nsystem: ...\\nuser: ..."}\n'
        f"\nTool schema:\n{func_json}\n"
    )

def build_generation_prompt_mode4_multiple(tool_list, memory_guidance=None):
    tools_json = json.dumps(tool_list, ensure_ascii=False, indent=2)
    guidance_text = ""
    if memory_guidance:
        guidance_text = (
            f"\nMemory Guidance:\n"
            f"Summary: {memory_guidance.get('summary','')}\n"
            f"Categories: {json.dumps(memory_guidance.get('categories', []), ensure_ascii=False)}\n"
            f"Guidance: {memory_guidance.get('guidance','')}\n"
        )
    return (
        "Your task is to generate 1 meaningful multi-turn dialogue that uses ALL the given tools. The dialogue should contain multiple user tasks; some tools may be called at least twice.\n"
        f"{guidance_text}\n"
        "Requirements:\n"
        "1) Write alternating lines (3 to 6 turns).\n"
        "2) Ensure each intended tool call is supported by explicit parameter values.\n"
        "3) In the FINAL user turn, explicitly list all required parameter values for EACH tool with concrete numbers.\n"
        "Return JSON:\n"
        '{"question":"user: ...\\nsystem: ...\\nuser: ..."}\n'
        f"\nTools schema:\n{tools_json}\n"
    )


def _last_user_utterance(text):
    if not isinstance(text, str):
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        lower = ln.lower()
        if lower.startswith("user:"):
            return ln.split(":", 1)[1].strip() or ln  # 去掉前缀，保留内容
    return None

PROMPT_BUILDERS = {
    "mode1": {"single": build_generation_prompt_mode1_single, "multiple": build_generation_prompt_mode1_multiple},
    "mode2": {"single": build_generation_prompt_mode2_single, "multiple": build_generation_prompt_mode2_multiple},
    "mode3": {"single": build_generation_prompt_mode3_single, "multiple": build_generation_prompt_mode3_multiple},
    "mode4": {"single": build_generation_prompt_mode4_single, "multiple": build_generation_prompt_mode4_multiple},
}

def build_generation_prompt(tool_list, is_multi, memory_guidance=None, mode="mode1"):
    registry = PROMPT_BUILDERS.get(mode) or PROMPT_BUILDERS["mode1"]
    builder = registry["multiple" if is_multi else "single"]
    if is_multi:
        return builder(tool_list, memory_guidance)
    else:
        return builder(tool_list[0], memory_guidance)


def generate_examples_for_tool_with_agents(tool, num_questions=5, model_name="gemini-2.5-pro", mode="mode1"):
    func_field = tool.get("function")
    tool_list = func_field if isinstance(func_field, list) else [func_field]
    is_multi = len(tool_list) > 1
    results = []
    memory_history = []
    tool_id = tool.get("id", "unknown")
    for i in range(num_questions):
        memory_guidance = memory_agent_summarize(memory_history, model_name=model_name) if memory_history else None
        candidates = []
        for _ in range(2):
            try:
                gen_prompt = build_generation_prompt(tool_list, is_multi, memory_guidance, mode=mode)
                raw = single_search(gen_prompt, model_name=model_name, temperature=1.0, json_mode=True)
                content = parse_search_result(raw)
                item = json.loads(clean_raw_json(content["response"]))
                q = item["question"] if isinstance(item, dict) else item[0]["question"]
                if is_multi:
                    calls = function_agent_extract_multiple(q, tool_list, model_name=model_name)
                else:
                    single_call = function_agent_extract_single(q, tool_list[0], model_name=model_name)
                    calls = [single_call] if single_call else []
                if not calls:
                    continue
                candidate = {"question": q, "function_calls": calls}
                candidates.append(candidate)
            except Exception as e:
                print(f"⚠️ Candidate generation error: {e}")
                continue
        if len(candidates) < 2:
            print("⚠️ Not enough candidates this round; skipping.")
            continue
        judge = judge_agent_choose(candidates[0], candidates[1], model_name=model_name)
        winner_flag = judge.get("winner", "A")
        if winner_flag == "A":
            winner = candidates[0]
        elif winner_flag == "B":
            winner = candidates[1]
        else:
            print("⚠️ Both candidates invalid; skipping round.")
            continue
        try:
            cat = memory_agent_classify(winner["question"], model_name=model_name)
        except Exception:
            cat = "unknown"
        memory_history.append({
            "question": winner["question"],
            "function_calls": winner["function_calls"],
            "category": cat
        })
        results.append(winner)
        print(f"✅ Generated example {len(results)}/{num_questions} for tool {tool_id}")
    return results

def main():
    args = parse_args()
    tools = json.load(open(args.input, "r", encoding="utf-8"))
    if isinstance(tools, dict):
        tools = [tools]
    elif not isinstance(tools, list):
        raise ValueError("Input JSON must be an array or an object.")
    all_examples, failed = [], []
    for idx, tool in enumerate(tools):
        print(f"\n🔄 Tool {idx+1}/{len(tools)}: {tool.get('id', f'idx_{idx}')}")
        try:
            examples = generate_examples_for_tool_with_agents(
                tool,
                num_questions=args.num_questions,
                model_name=args.model,
                mode=args.mode,
            )
            all_examples.extend(examples)
            print(f"✅ {len(examples)} kept for tool {tool.get('id')}.")
        except Exception as e:
            print(f"❌ Failed tool {tool.get('id')}: {e}")
            failed.append(tool.get('id'))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"examples": all_examples}, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Done. {len(all_examples)} examples saved to {args.output}")
    if failed:
        print(f"⚠️ Failed tools: {failed}")

if __name__ == '__main__':
    main()
