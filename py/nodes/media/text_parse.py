from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

JsonRepair = Callable[[str], str]

def _repair_llm_json(
    text: str, repairs: list[JsonRepair] | None = None
) -> str:

    repaired = text
    for repair in repairs or ():
        repaired = repair(repaired)

    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired

def _require_object_array(parsed) -> list[dict] | None:
    if not isinstance(parsed, list):
        return None
    if not parsed:
        return None
    if not all(isinstance(item, dict) for item in parsed):
        raise ValueError("JSON 配列の各要素は object である必要があります")
    return parsed

def _try_parse_object_array(text: str) -> list[dict] | None:

    decoder = json.JSONDecoder()
    try:
        parsed, _end = decoder.raw_decode(text.lstrip())
    except json.JSONDecodeError:
        return None
    return _require_object_array(parsed)

def _scan_object_arrays(text: str) -> list[dict]:

    best: list[dict] = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", text):
        try:
            parsed, _end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        try:
            object_array = _require_object_array(parsed)
        except ValueError:
            continue
        if object_array is not None and len(object_array) > len(best):
            best = object_array
    return best

def extract_json_array(
    text: str, *, repairs: list[JsonRepair] | None = None
) -> list[dict]:
    stripped = _repair_llm_json(text.strip(), repairs)

    for candidate in (stripped, text.strip()):
        object_array = _try_parse_object_array(candidate)
        if object_array is not None:
            return object_array

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
    if fence:
        fenced = _repair_llm_json(fence.group(1).strip(), repairs)
        object_array = _try_parse_object_array(fenced)
        if object_array is not None:
            return object_array

    best = _scan_object_arrays(stripped)
    if best:
        return best

    raise ValueError(f"JSON 配列が見つからない:\n{stripped[:500]}")

def find_last_json_array(text: str, *, allow_str: bool = False) -> list | None:
    # モデルは 1 つ目を出したあと「修正して再出力します」と続けることがあるので、
    # extract_json_array と違い**最後の**配列を採り、失敗しても raise せず None を返す。
    # allow_str=True で文字列の配列も受ける (既定は辞書の配列だけ)。

    fences = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    for raw in reversed(fences):
        try:
            got = json.loads(raw)
        except ValueError:
            continue
        if isinstance(got, list):
            return got

    found: list[list] = []
    for start in (i for i, c in enumerate(text) if c == "["):
        depth = 0
        in_str = False
        esc = False
        for j in range(start, len(text)):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    try:
                        got = json.loads(text[start:j + 1])
                    except ValueError:
                        break

                    if isinstance(got, list) and (
                            not got or isinstance(got[0], dict)
                            or (allow_str and isinstance(got[0], str))):
                        found.append(got)
                    break
    return found[-1] if found else None

def find_json_object(text: str) -> dict | None:
    # extract_json_object と違い、失敗しても raise せず None を返す。

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        try:
            got = json.loads(fence.group(1))
            if isinstance(got, dict):
                return got
        except ValueError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            got = json.loads(text[start:end + 1])
            if isinstance(got, dict):
                return got
        except ValueError:
            pass
    return None


def find_last_json_object(text: str) -> dict | None:
    """Return the last fenced JSON object, or the enclosing raw object."""

    blocks = [
        match.group(1)
        for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    ]
    if not blocks:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            blocks = [text[start : end + 1]]
    for raw in reversed(blocks):
        try:
            parsed = json.loads(raw)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None

def extract_json_object(text: str) -> dict[str, Any]:

    if "{" not in text:
        raise RuntimeError(f"no JSON object in response: {text[:400]}")

    last_unbalanced_start: int | None = None
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_str = False
        esc = False
        balanced_end: int | None = None
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    balanced_end = i + 1
                    break
        if balanced_end is not None:
            candidate = text[start:balanced_end]
            try:
                # strict=False: accept a literal line break / tab inside a
                # string value. Long planner outputs (measured 2026-08-19: a
                # 50-page comic scenario, $1.2 per call) occasionally wrap a
                # Japanese sentence mid-string; the content is intact and
                # rejecting the whole answer for it only burns the call.
                data = json.loads(candidate, strict=False)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(data, dict):
                    return data

        elif last_unbalanced_start is None:
            last_unbalanced_start = start

        next_start = balanced_end if balanced_end is not None else start + 1
        start = text.find("{", next_start)

    if last_unbalanced_start is not None:
        raise RuntimeError(
            f"unbalanced JSON in response: "
            f"{text[last_unbalanced_start:last_unbalanced_start + 400]}"
        )
    raise RuntimeError(f"no valid JSON object in response: {text[:400]}")
