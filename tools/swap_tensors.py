#!/usr/bin/env python3
"""Write a copy of a JunkInfer GGUF in which every tensor whose name matches REGEX is taken from a donor GGUF of the
same model (same names / layout, any ggml type), e.g. the dense projections from a mainline quantization while the
EXL3 experts stay. The EXL3 scale vectors (<name>.suh / .svh) of replaced weights are dropped.

usage: swap_tensors.py BASE_GGUF DONOR_GGUF OUT_GGUF REGEX
example REGEX (dense projections): 'blk\\.\\d+\\.(attn_(q|k|v|output|qkv|gate)|ssm_out|ffn_(gate|up|down)_shexp)\\.weight'
"""
import os, re, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/junkinfer/engine/gguf-py"))
import gguf

base, donor, out, pat = sys.argv[1:5]
rx = re.compile(pat)
rb, rd = gguf.GGUFReader(base), gguf.GGUFReader(donor)
dt = {t.name: t for t in rd.tensors}
arch = rb.fields["general.architecture"].contents()
w = gguf.GGUFWriter(out, arch)
skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
for f in rb.fields.values():
    if f.name in skip:
        continue
    vt = f.types[0]
    sub = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, f.contents(), vt, sub_type=sub)
swapped = {t.name for t in rb.tensors if rx.fullmatch(t.name)}
missing = [n for n in swapped if n not in dt]
assert not missing, f"donor lacks {missing[:5]}"
drop = {n[:-len(".weight")] + s for n in swapped for s in (".suh", ".svh")}
plan = [dt[t.name] if t.name in swapped else t for t in rb.tensors if t.name not in drop]
for t in plan:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, raw_dtype=t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in plan:
    w.write_tensor_data(np.asarray(t.data))
w.close()
types = {}
for n in swapped: types[dt[n].tensor_type.name] = types.get(dt[n].tensor_type.name, 0) + 1
print(f"wrote {out}: {len(swapped)} tensors swapped ({types}), {len(drop & {t.name for t in rb.tensors})} scale vectors dropped")
