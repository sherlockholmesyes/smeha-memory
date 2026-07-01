# SPDX-License-Identifier: Apache-2.0
"""Fact extraction: raw text -> atomic (subject, predicate, object) triples via a LOCAL ollama LLM
(the text never leaves the machine). Feeds a TemporalFactStore (auto-supersede) or a graph.

Robust JSON parsing: handles ollama `format=json`, the {"facts":[...]} / {"triples":[...]} shapes,
the column-oriented {"subject":[...],"predicate":[...],"object":[...]} some models emit, list-of-lists,
and a bracket-salvage fallback. Optional dependency: a running ollama with the chosen model.
"""
from __future__ import annotations

import json
import re
import urllib.request

_PROMPT = (
    'Extract the facts as JSON: {{"facts": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}\n'
    'Rules:\n'
    '1. Resolve pronouns/coreference to the MAIN named entity ("it"/"its X"/"the X" -> the company/subject).\n'
    '2. Predicate = a SHORT consistent verb-relation (uses, deploys_on, written_in, stores_in, has),'
    ' NEVER a bare preposition like "to"/"from".\n'
    '3. For a change/migration, output ONLY the NEW state, with the SAME relation the old state uses'
    ' ("migrated deployment from AWS to GCP" -> subject=Acme, predicate=deploys_on, object=GCP).\n'
    '4. One object per fact; short noun phrases; only facts explicitly stated.\n'
    'Example -> {{"facts": [{{"subject": "Acme", "predicate": "deploys_on", "object": "GCP"}}]}}\n'
    'Output JSON only.\n\nText: {text}\n\nJSON:'
)


def extract_facts(text, model="mistral:7b", host="http://localhost:11434", timeout=120):
    """Return a list of (subject, predicate, object) tuples extracted from `text`."""
    req = urllib.request.Request(
        host + "/api/generate",
        data=json.dumps({"model": model, "prompt": _PROMPT.format(text=text),
                         "stream": False, "format": "json",
                         "options": {"temperature": 0}}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return _parse_triples(json.load(r).get("response", ""))


def _parse_triples(s):
    try:
        data = json.loads(s)
    except Exception:
        m = re.search(r"\[.*\]", s, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []
    if isinstance(data, dict):                      # {"facts":[...]} / {"triples":[...]} / column / single
        low = {k.lower(): v for k, v in data.items()}
        # column-oriented {subject:[...],predicate:[...],object:[...]} (some models emit this)
        if isinstance(low.get("object"), list) and "subject" in low:
            subs = low["subject"] if isinstance(low["subject"], list) else [low["subject"]]
            preds = low.get("predicate") if isinstance(low.get("predicate"), list) else [low.get("predicate")]
            cols = []
            for i, o in enumerate(low["object"]):
                s_ = subs[i] if i < len(subs) else (subs[-1] if subs else None)
                p_ = preds[i] if i < len(preds) else (preds[-1] if preds else None)
                if s_ and p_ and o:
                    cols.append((str(s_).strip(), str(p_).strip(), str(o).strip()))
            if cols:
                return cols
        for k in ("facts", "triples", "results", "data"):
            if isinstance(data.get(k), list):
                data = data[k]
                break
        else:
            data = [data]
    out = []
    for t in (data if isinstance(data, list) else []):
        if isinstance(t, (list, tuple)) and len(t) >= 3:
            out.append((str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()))
        elif isinstance(t, dict):
            s_, p_, o_ = (t.get("subject") or t.get("s"), t.get("predicate") or t.get("p"),
                          t.get("object") or t.get("o"))
            if s_ and p_ and o_:
                out.append((str(s_).strip(), str(p_).strip(), str(o_).strip()))
    return [(a, b, c) for a, b, c in out if a and b and c]


def extract_into_temporal(text, tstore, **kw):
    """Extract facts from text and assert each into a TemporalFactStore (auto-supersede)."""
    facts = extract_facts(text, **kw)
    for s, p, o in facts:
        tstore.add_fact(s, p, o, source="extractor")
    return facts
