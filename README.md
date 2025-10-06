---
title: Quantized_lang._Translator
app_file: app.py
sdk: gradio
sdk_version: 5.48.0
---
# NLLB-FB Language Translator (Quantized, CPU-Friendly)

This project provides a quantized, CPU-optimized version of the NLLB (No Language Left Behind) Facebook language translation model. It enables very fast inference on CPUs for translating between a wide variety of languages.

## Features

- **Quantized Model:** Reduced model size for efficient CPU usage.
- **Fast Inference:** Optimized for low-latency translation on standard CPUs.
- **Multi-language Support:** Translate between many language pairs.
- **Easy Integration:** Simple API for batch and single-sentence translation.

## Usage

1. **Install dependencies:**
    ```bash
    pip install torch transformers
    ```

2. **Run the Gradio app:**
    ```bash
    python app.py
    ```
  ## Hugging Face Space
  [Link Text](https://huggingface.co/spaces/RohanAi/Quantized_lang._Translator)

## Supported Languages

See the [NLLB-200 language list](https://github.com/facebookresearch/fairseq/tree/main/examples/nllb) for all supported languages.

## References

- [NLLB: No Language Left Behind](https://ai.facebook.com/research/no-language-left-behind/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/model_doc/nllb)
