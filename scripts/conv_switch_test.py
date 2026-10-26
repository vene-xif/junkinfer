#!/usr/bin/env python3
"""Conversation-switch regression test: A (long), B (unrelated, long), A again x2 through the prompt cache.
Before engine fix 2026-10-26, with MTP on, the switch back to A failed (HTTP 500 "failed to process speculative
batch": the MTP draft's deferred rows of B replayed into A) and later requests on A ran on a damaged state
(tool calls emitted as text). Pass = no HTTP error and A's answers equal after the switch.
usage: conv_switch_test.py [base_url]"""
import json, sys, urllib.request, urllib.error
URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080") + "/completion"
sysp = "You are a careful assistant. " + " ".join(f"Rule {i}: answer precisely and cite the rule number when relevant." for i in range(900))
A = f"<|im_start|>system\n{sysp}<|im_end|>\n<|im_start|>user\nWhat is 17 * 23? Answer with the number only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
B = "<|im_start|>user\nSummarize in one sentence:\n" + "The history of lighthouses spans more than two thousand years. " * 600 + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
def run(p, label):
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps({"prompt": p, "n_predict": 40, "temperature": 0, "cache_prompt": True}).encode(), {"Content-Type": "application/json"}), timeout=1800))
    except urllib.error.HTTPError as e:
        print(f"FAIL {label}: HTTP {e.code} {e.read()[:150]!r}"); return None
    t = d["timings"]; print(f"     {label}: prompt_n {t.get('prompt_n')} cache_n {t.get('cache_n')}: {d['content'][:60]!r}"); return d["content"]
a0 = run(A, "A"); run(B, "B"); a1 = run(A, "A after B"); a2 = run(A, "A again")
ok = None not in (a0, a1, a2) and a1 == a2 and "391" in (a1 or "")
print("ALL PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
