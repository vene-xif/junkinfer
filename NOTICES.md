# Third-party notices

JunkInfer is Apache-2.0. It uses or adapts code from the projects below. Each is under the
MIT License; their copyright notices are kept with any code taken from them.

| project | license | used for |
|---|---|---|
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) | MIT, (c) The ggml authors | base engine, GGUF, kernels, server |
| [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | MIT, (c) The ggml authors | CPU kernels, fused MoE, trellis (KT) quant ideas |
| [beellama.cpp](https://github.com/Anbeeld/beellama.cpp) | MIT, (c) The ggml authors, Anbeeld | KVarN KV-cache implementation |
| [exllamav3](https://github.com/turboderp-org/exllamav3) | MIT, (c) 2025 Turboderp | EXL3 / QTIP trellis format and codebooks |
| [Strata](https://github.com/Niko1221/Strata) | MIT, (c) 2026 Niko1221 and contributors | hybrid CPU/GPU expert scheduling design |
| [BELLS](https://github.com/DGuckert/llama.cpp-BELLS) | MIT (llama.cpp fork) | expert cache and prompt staging design |

Algorithms from papers (QTIP, KVarN, Quest, MInference, XAttention) are credited where implemented.
KVarN is described by Huawei researchers; this project has not checked for patents on the method.
