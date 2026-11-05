"""Strict output contract shared by teacher generation and label ingestion.

Grammar constraints fix transport syntax only. They do not certify visual truth.
"""
import json

OUTPUT_VERSION = "teacher_json_v2"
ASPECT_NAMES = ("mission_usefulness","evidence_clarity","additional_information","condition_specificity")


def answer_schema(task):
    if task not in ("candidate", "prefix", "control", "replay", "tier"):
        raise ValueError("unknown teacher task")
    key = "choice" if task in ("candidate", "prefix", "control") else "value"
    choices = (["A", "B", "tie", "uncertain"] if key == "choice" else
               ["low", "neutral", "high", "uncertain"] if task == "replay" else
               ["Critical", "High", "Medium", "Low", "uncertain"])
    properties = {
        key: {"type": "string", "enum": choices},
        "evidence": {"type": "string", "maxLength": 240},
        "unknowns": {"type": "array", "items": {"type": "string", "maxLength": 120}, "maxItems": 4},
    }
    required = [key, "evidence", "unknowns"]
    if task == "candidate":
        # Fixed objects avoid the upstream grammar parser's nullable-array bug.
        rating = {"type":"object", "properties":{
            name:{"type":"string","enum":["0","1","2","3","4","unknown"]}
            for name in ASPECT_NAMES}, "required":list(ASPECT_NAMES), "additionalProperties":False}
        properties["aspects"] = {
            "type": "object", "properties": {"A": rating, "B": rating},
            "required": ["A", "B"], "additionalProperties": False}
        required.append("aspects")
    return {"type": "object", "properties": properties,
            "required": required, "additionalProperties": False}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse_answer(text, task):
    """Accept one JSON object, optionally fenced; never guess a malformed label."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("unclosed JSON fence")
        text = "\n".join(lines[1:-1]).strip()
    def invalid_constant(value):
        raise ValueError("nonfinite JSON constant")
    row = json.loads(text, object_pairs_hook=_unique_object,
                     parse_constant=invalid_constant)
    schema = answer_schema(task)
    if not isinstance(row, dict) or set(row) != set(schema["required"]):
        raise ValueError("teacher keys do not match task schema")
    key = "choice" if "choice" in row else "value"
    if row[key] not in schema["properties"][key]["enum"]:
        raise ValueError("invalid teacher category")
    if not isinstance(row["evidence"], str) or len(row["evidence"]) > 240:
        raise ValueError("invalid evidence")
    unknowns = row["unknowns"]
    if not isinstance(unknowns, list) or len(unknowns) > 4 or any(
            not isinstance(item, str) or len(item) > 120 for item in unknowns):
        raise ValueError("invalid unknowns")
    if task == "candidate":
        aspects = row["aspects"]
        if not isinstance(aspects, dict) or set(aspects) != {"A", "B"}:
            raise ValueError("invalid aspect keys")
        normalized = {}
        for key,value in aspects.items():
            if not isinstance(value,dict) or set(value)!=set(ASPECT_NAMES):
                raise ValueError("invalid aspect dimensions")
            ordered=[value[name] for name in ASPECT_NAMES]
            if any(type(item) is not str or item not in ("0","1","2","3","4","unknown") for item in ordered):
                raise ValueError("invalid aspect ratings")
            normalized[key]=None if all(item=="unknown" for item in ordered) else [None if item=="unknown" else int(item) for item in ordered]
        row["aspects"]=normalized

    return row


def prefix_factory(tokenizer):
    """Build static vocabulary data once; every answer gets an independent parser."""
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn, build_token_enforcer_tokenizer_data)
    data=build_token_enforcer_tokenizer_data(tokenizer)
    def constraint(task):
        return build_transformers_prefix_allowed_tokens_fn(data,JsonSchemaParser(answer_schema(task)))
    return constraint


def prefix_constraint(tokenizer, task):
    """One-off convenience for tests; a teacher session uses prefix_factory once."""
    return prefix_factory(tokenizer)(task)
