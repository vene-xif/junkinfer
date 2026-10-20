import json, os, sys
sys.path.insert(0, os.path.expanduser("~/junkinfer/engine/gguf-py"))
import gguf
M = "/mnt/ssd/models/MiniCPM-V-4.7-35B-A3B-EXL3-SAGE-3.74bpw"
tj = json.load(open(f"{M}/tokenizer.json"))
vocab = dict(tj["model"]["vocab"])
for a in tj.get("added_tokens", []): vocab[a["content"]] = a["id"]
inv = {v: k for k, v in vocab.items()}
print("minicpm: model vocab", len(tj["model"]["vocab"]), "added", len(tj.get("added_tokens", [])), "max id", max(inv))
r = gguf.GGUFReader("/mnt/ssd/models/Qwen3.6-35B-A3B-exl3-4.00bpw.gguf")
f = r.fields["tokenizer.ggml.tokens"]
toks = [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]
print("qwen gguf tokens", len(toks))
diff = [(i, toks[i], inv.get(i)) for i in range(min(len(toks), max(inv) + 1)) if inv.get(i) != toks[i]]
print("differences", len(diff)); print(diff[:15])
print("merges same:", tj["model"].get("merges", [])[:3])
tc = json.load(open(f"{M}/tokenizer_config.json")); print("eos", tc.get("eos_token"), "pad", tc.get("pad_token"))
for k in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id", "tokenizer.ggml.bos_token_id"):
    if k in r.fields: print(k, r.fields[k].contents())
