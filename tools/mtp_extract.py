#!/usr/bin/env python3
"""Extract the MTP (nextn) sidecar from a full qwen35moe GGUF: all metadata, output head, token embedding and the
nextn block (blk.<n_layer-1>.*), tensors copied unchanged. usage: mtp_extract.py full.gguf out.gguf"""
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType

src, dst = sys.argv[1], sys.argv[2]
r = GGUFReader(src)
arch = bytes(r.fields["general.architecture"].parts[r.fields["general.architecture"].data[0]]).decode()
nblk = int(r.fields[f"{arch}.block_count"].parts[r.fields[f"{arch}.block_count"].data[0]][0])
mtp_blk = nblk - 1
w = GGUFWriter(dst, arch, use_temp_file=True)
for f in r.fields.values():
    if f.name.startswith("GGUF.") or f.name == "general.architecture":
        continue
    vt = f.types[0]
    if vt == GGUFValueType.ARRAY:
        et = f.types[1]
        if et == GGUFValueType.STRING:
            vals = [bytes(f.parts[i]).decode("utf-8", errors="surrogateescape") for i in f.data]
        else:
            vals = [f.parts[i].tolist()[0] for i in f.data]
        w.add_key_value(f.name, vals, vt, sub_type=et)
    elif vt == GGUFValueType.STRING:
        w.add_key_value(f.name, bytes(f.parts[f.data[0]]).decode("utf-8", errors="surrogateescape"), vt)
    else:
        w.add_key_value(f.name, f.parts[f.data[0]].tolist()[0], vt)
keep = [t for t in r.tensors if t.name in ("output.weight", "output_norm.weight", "token_embd.weight")
        or t.name.startswith(f"blk.{mtp_blk}.")]
for t in keep:
    data = np.asarray(t.data)   # quantized / BF16: byte-shaped rows; the writer converts byte shapes to elements
    w.add_tensor(t.name, data, raw_shape=list(data.shape), raw_dtype=t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(progress=False); w.close()
print("wrote", dst, len(keep), "tensors, mtp block", mtp_blk)
