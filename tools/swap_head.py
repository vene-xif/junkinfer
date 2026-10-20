#!/usr/bin/env python3
"""Write a copy of a JunkInfer GGUF whose LM head (output.weight) is taken from another GGUF of the same model
(e.g. an MTP sidecar quantized with --output-tensor-type iq4_nl); EXL3 head scales (output.suh / .svh) are dropped.

usage: swap_head.py BASE_GGUF HEAD_GGUF OUT_GGUF
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.expanduser("~/junkinfer/engine/gguf-py"))
import gguf

base, headf, out = sys.argv[1:4]
rb = gguf.GGUFReader(base)
rh = gguf.GGUFReader(headf)
head = [t for t in rh.tensors if t.name == "output.weight"][0]
arch = rb.fields["general.architecture"].contents()
w = gguf.GGUFWriter(out, arch)
skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
for f in rb.fields.values():
    if f.name in skip:
        continue
    vt = f.types[0]
    sub = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, f.contents(), vt, sub_type=sub)
drop = {"output.suh", "output.svh"}
plan = []
for t in rb.tensors:
    if t.name in drop:
        continue
    src = head if t.name == "output.weight" else t
    plan.append(src)
for t in plan:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, raw_dtype=t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in plan:
    w.write_tensor_data(np.asarray(t.data))
w.close()
print(f"wrote {out}: head {head.tensor_type.name} {list(head.shape)}")
