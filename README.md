# speculative-decoding-T4-GPU

# Speculative Decoding on a T4 GPU

Tested when speculative decoding actually speeds up generation, using Qwen2.5-3B-Instruct as the target model on a single NVIDIA T4 (Kaggle), with Hugging Face and vLLM.

## Results (tokens/sec, greedy, 200 max tokens, FP16)

| Engine | Prompt | Setup | tok/s | Speed-up |
|---|---|---|---|---|
| Hugging Face | General | 3B alone | 21 | 1.0x |
| Hugging Face | General | 3B + 0.5B draft | 13 | 0.6x |
| vLLM | General | 3B alone | 36.9 | 1.0x |
| vLLM | General | 3B + 0.5B draft, 5 tokens | 26.2 | 0.71x |
| vLLM | General | 3B + 0.5B draft, 2 tokens | 46.6 | 1.26x |
| vLLM | Invoice extraction | 3B alone | 24 | 1.0x |
| vLLM | Invoice extraction | n-gram, 4 tokens | 43.6 | **1.8x** |
| vLLM | Invoice extraction | n-gram, 8 tokens | 42.5 | 1.77x |

## Key findings
- **Engine overhead matters first.** vLLM alone was 1.8x faster than Hugging Face, because of CUDA graphs and lower per-token overhead.
- **Draft length is a trade-off.** With a 0.5B draft, 5 guessed tokens was slower (0.71x) but 2 was faster (1.26x): long guesses get rejected and waste work.
- **N-gram drafting wins on extraction.** For outputs that copy from the input, like invoice fields, prompt-lookup drafting gave 1.8x with no draft model at all.
- **4 speculative tokens was the sweet spot**; 8 added no gain.

## Takeaway
Speculative decoding is not free. It pays off when draft tokens are cheap and mostly accepted, which is why n-gram drafting suits document-intelligence workloads.

## How to run
Kaggle, GPU T4, internet on. Install vLLM, then run each cell, restarting the kernel between runs to free GPU memory.
