# K128 SM70 grouped-MoE local probe

Keep this experiment outside the engine. Reuse the existing bounded E85 IQ3_XXS
and IQ3_S samples, original packed decoder and Volta MMA fragment mapping.

The current TC kernel stages 16 K values at a time using 96 of 256 threads,
with two CTA barriers for every 16 K. Stage 128 K cooperatively, then consume
the same eight 16-K fragments in the same accumulation order. This removes
seven eighths of the barriers and gives all threads staging work. Use one
shared buffer; no new weight format, arithmetic or persistent device cache.

Gate: full-output bit equality against the existing TC kernel with identical
work list, changed-input replay and output canaries; retain stock-CUDA and
selected original-weight FP64 error reports. Screen grid sizes 80/160/320 with
CUDA-event timings. A local win is not an engine integration or model-quality
claim. No GPU run while another owner has the six-card fixture window.
