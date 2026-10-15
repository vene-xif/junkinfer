#!/usr/bin/env python3
"""Build the fixed, tokenized evaluation corpus for JunkInfer quality gates.

Outputs (JTOK v1, see junk-ref.cpp for the format):
  S1_wiki.jtok   48 rows x 2048 tokens of wikitext-2 test, scored on positions [1024, 2047)
  S2_code.jtok   48 rows x 2048 tokens of llama.cpp source ("eval" half of the files)
  L_code.jtok    3 rows: the first 32768 / 131072 / 262144 tokens of the eval code stream, scored on the last 1024
  L_prose.jtok   same, from War and Peace (Project Gutenberg #2600)
  calib_base.txt imatrix calibration text: wikitext-2 TRAIN + the "calib" half of the source files
  manifest.json  input hashes, token counts, row layout

Everything is deterministic for fixed inputs.
"""
import argparse, hashlib, json, os, struct, sys

from tokenizers import Tokenizer

CODE_DIRS = ["src", "ggml/src", "common", "tools", "examples"]
CODE_EXT = {".c", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".py", ".comp"}
MAX_FILE_BYTES = 400_000
SHORT_ROWS, SHORT_LEN, SHORT_A = 48, 2048, 1024
LONG_LENS = [32768, 131072, 262144]
LONG_SCORED = 1024
CODE_SHORT_OFFSET = 300_000


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_jtok(path, rows):
    """rows: list of (tokens, a, b)"""
    with open(path, "wb") as f:
        f.write(b"JTOK")
        f.write(struct.pack("<II", 1, len(rows)))
        for toks, a, b in rows:
            assert 0 <= a < b <= len(toks) - 1, (a, b, len(toks))
            f.write(struct.pack("<III", len(toks), a, b))
            f.write(struct.pack(f"<{len(toks)}i", *toks))


def code_files(root):
    out = []
    for d in CODE_DIRS:
        base = os.path.join(root, d)
        for dp, dn, fn in os.walk(base):
            dn.sort()
            parts = dp.split(os.sep)
            if "vendor" in parts or "junk-ref" in parts or "build" in parts:
                continue
            for n in sorted(fn):
                p = os.path.join(dp, n)
                if os.path.splitext(n)[1] not in CODE_EXT:
                    continue
                if os.path.getsize(p) > MAX_FILE_BYTES:
                    continue
                out.append(os.path.relpath(p, root))
    out.sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--wiki-dir", required=True, help="dir with wiki.test.raw and wiki.train.raw")
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--prose", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-only", action="store_true", help="validate inputs, write nothing")
    args = ap.parse_args()

    wiki_test = os.path.join(args.wiki_dir, "wiki.test.raw")
    wiki_train = os.path.join(args.wiki_dir, "wiki.train.raw")
    for p in [args.tokenizer, wiki_test, wiki_train, args.prose]:
        if not os.path.isfile(p):
            sys.exit(f"missing input: {p}")
    files = code_files(args.code_root)
    if len(files) < 200:
        sys.exit(f"too few code files under {args.code_root}: {len(files)}")
    if args.check_only:
        print(f"prep_corpus: inputs OK ({len(files)} code files)")
        return

    tok = Tokenizer.from_file(args.tokenizer)
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    os.makedirs(args.out, exist_ok=True)

    # --- S1 wiki
    with open(wiki_test, encoding="utf-8") as f:
        wiki = enc(f.read())
    need = SHORT_ROWS * SHORT_LEN
    assert len(wiki) >= need, (len(wiki), need)
    s1 = [(wiki[i * SHORT_LEN:(i + 1) * SHORT_LEN], SHORT_A, SHORT_LEN - 1) for i in range(SHORT_ROWS)]
    write_jtok(os.path.join(args.out, "S1_wiki.jtok"), s1)

    # --- code streams, split by path hash
    eval_files, calib_files = [], []
    for rel in files:
        (eval_files if int(hashlib.sha1(rel.encode()).hexdigest(), 16) % 2 == 0 else calib_files).append(rel)
    code_eval = []
    for rel in eval_files:
        with open(os.path.join(args.code_root, rel), encoding="utf-8", errors="replace") as f:
            code_eval += enc(f"// ===== {rel} =====\n" + f.read() + "\n")
    need = CODE_SHORT_OFFSET + SHORT_ROWS * SHORT_LEN
    assert len(code_eval) >= max(need, LONG_LENS[-1] + 1), len(code_eval)
    s2 = [(code_eval[CODE_SHORT_OFFSET + i * SHORT_LEN:CODE_SHORT_OFFSET + (i + 1) * SHORT_LEN], SHORT_A, SHORT_LEN - 1)
          for i in range(SHORT_ROWS)]
    write_jtok(os.path.join(args.out, "S2_code.jtok"), s2)
    write_jtok(os.path.join(args.out, "L_code.jtok"),
               [(code_eval[:L], L - LONG_SCORED - 1, L - 1) for L in LONG_LENS])

    # --- long prose
    with open(args.prose, encoding="utf-8-sig") as f:
        txt = f.read()
    s = txt.find("*** START OF")
    e = txt.find("*** END OF")
    if s >= 0:
        s = txt.find("\n", s) + 1
    txt = txt[max(s, 0):e if e > 0 else len(txt)]
    prose = enc(txt)
    assert len(prose) >= LONG_LENS[-1] + 1, len(prose)
    write_jtok(os.path.join(args.out, "L_prose.jtok"),
               [(prose[:L], L - LONG_SCORED - 1, L - 1) for L in LONG_LENS])

    # --- imatrix calibration text (disjoint from eval: wiki TRAIN, the other half of the files)
    with open(wiki_train, encoding="utf-8") as f:
        wtrain = f.read()[:450_000]
    code_parts = []
    total = 0
    for rel in calib_files:
        with open(os.path.join(args.code_root, rel), encoding="utf-8", errors="replace") as f:
            t = f"// ===== {rel} =====\n" + f.read() + "\n"
        code_parts.append(t)
        total += len(t)
        if total > 450_000:
            break
    # interleave ~20k-char blocks so every imatrix chunk sees a mix
    wb = [wtrain[i:i + 20000] for i in range(0, len(wtrain), 20000)]
    code_txt = "".join(code_parts)
    cb = [code_txt[i:i + 20000] for i in range(0, len(code_txt), 20000)]
    mixed = []
    for i in range(max(len(wb), len(cb))):
        if i < len(wb): mixed.append(wb[i])
        if i < len(cb): mixed.append(cb[i])
    with open(os.path.join(args.out, "calib_base.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(mixed))

    manifest = {
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "wiki_test_sha256": sha256_file(wiki_test),
        "wiki_train_sha256": sha256_file(wiki_train),
        "prose_sha256": sha256_file(args.prose),
        "code_files_eval": len(eval_files), "code_files_calib": len(calib_files),
        "tokens": {"wiki_test": len(wiki), "code_eval": len(code_eval), "prose": len(prose)},
        "sets": {
            "S1_wiki": {"rows": SHORT_ROWS, "len": SHORT_LEN, "score": [SHORT_A, SHORT_LEN - 1]},
            "S2_code": {"rows": SHORT_ROWS, "len": SHORT_LEN, "score": [SHORT_A, SHORT_LEN - 1], "offset": CODE_SHORT_OFFSET},
            "L_code": {"lens": LONG_LENS, "scored_last": LONG_SCORED},
            "L_prose": {"lens": LONG_LENS, "scored_last": LONG_SCORED},
        },
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest["tokens"]))


if __name__ == "__main__":
    main()
