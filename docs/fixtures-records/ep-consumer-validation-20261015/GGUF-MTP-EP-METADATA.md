# Target and MTP expert tensor metadata

The checked file is:

`/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf`

The tensor directory was read with the source tree's `gguf.GGUFReader`; no inference model or tensor payload was loaded for this metadata check. The integrated file has 1256 tensors. Its `blk.48` nextn expert tensors are `Q8_0` (GGML type id 8), with 512 experts:

| Tensor | Shape `[K,N,E]` | GGUF file offset | Expert stride |
|---|---:|---:|---:|
| `blk.48.ffn_gate_exps.weight` | `[2560,640,512]` | `76785953216` | `1740800` bytes |
| `blk.48.ffn_up_exps.weight` | `[2560,640,512]` | `77684236736` | `1740800` bytes |
| `blk.48.ffn_down_exps.weight` | `[640,2560,512]` | `75892922816` | `1740800` bytes |

`ReaderTensor.data_offset` already includes the GGUF data-section base; it must not be added to `GGUFReader.data_offset` a second time. The dense nextn head tensors are also Q8_0 (`blk.48.nextn.eh_proj.weight`, `hc_head_down`, and `hc_head_up`), but they are not `MUL_MAT_ID` expert banks. `tensors.json` includes the three routed Q8_0 banks in the fixture. They are assigned the independent `nextn` owner-count input so tests do not infer the draft slice from the target slice.

The seven target expert formats come from `fixtures/volta-tc-real/manifest.json`: `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, and `Q2_0`.
