import sys, os, numpy as np
sys.path.insert(0, os.path.expanduser("~/junkinfer/tools/exl3")); sys.path.insert(0, os.path.expanduser("~/junkinfer/engine/gguf-py"))
import exl3_ref, gguf
M = "/mnt/ssd/models/MiniCPM-V-4.7-35B-A3B-EXL3-SAGE-3.74bpw"
idx = exl3_ref.st_index(M)
r = gguf.GGUFReader(os.path.expanduser("~/models/ssd-overflow/Qwen3.6-35B-IQ4_NL.gguf"))
for il, proj, gname in [(0, "gate_proj", "ffn_gate_exps"), (0, "down_proj", "ffn_down_exps"), (21, "up_proj", "ffn_up_exps")]:
    t = [t for t in r.tensors if t.name == f"blk.{il}.{gname}.weight"][0]
    for e in (0, 77):
        pre = f"model.language_model.layers.{il}.mlp.experts.{e}.{proj}"
        bits = exl3_ref.st_read(M, idx, pre + ".trellis").shape[-1] // 16
        W, _ = exl3_ref.full_weight(M, pre)
        n_out, n_in = t.shape[1], t.shape[0]
        per = t.data.shape[0] // 256 if t.data.ndim > 1 else None
        raw = np.asarray(t.data).reshape(256, -1)[e]
        G = gguf.quants.dequantize(raw, t.tensor_type).reshape(-1, n_in).T
        cos = (W * G).sum() / np.sqrt((W * W).sum() * (G * G).sum())
        print(f"layer {il} expert {e} {proj} {bits}-bit: cosine vs Qwen3.6 IQ4_NL {cos:.4f}")
