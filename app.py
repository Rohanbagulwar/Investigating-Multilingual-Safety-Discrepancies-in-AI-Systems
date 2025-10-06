from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from sacremoses import MosesPunctNormalizer
from flores import code_mapping
import gradio as gr
import platform

device = "cpu" if platform.system() == "Windows" else "cuda"
MODEL_DIR ="RohanAi/nllb_quantized"
# MODEL_DIR = "./nllb-600M-quantized"

# 8-bit quantization for GPU
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if device == "cuda":
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_DIR, device_map="auto", quantization_config=bnb_config
    )
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)

punct_normalizer = MosesPunctNormalizer(lang="en")
print("device is ",device)
# Language mapping
langs = {
    "Hindi": "hin_Deva",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "German": "deu_Latn",
    "Arabic": "arb_Arab"
}

def translate(text: str, src_lang: str, tgt_lang: str):
    src_code = code_mapping[src_lang]   # e.g. "English" -> "eng_Latn"
    tgt_code = code_mapping[tgt_lang]
    print('source lang code ',src_code)   # e.g. "Hindi"   -> "hin_Deva"

    tokenizer.src_lang = src_code
    tokenizer.tgt_lang = tgt_code

    # Normalize punctuation
    text = punct_normalizer.normalize(text)

    # Encode & generate
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),  # use FLORES code
        # max_length=len(inputs.input_ids[0]) + 150,  # dynamic max length
        num_beams=3,       # CPU-friendly greedy decoding
        no_repeat_ngram_size=2,  # small repetition control
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


langs = list(code_mapping.keys())

iface = gr.Interface(
    fn=translate,
    inputs=[gr.Textbox(lines=10, label="Input Text"),
            gr.Dropdown(langs, label="Source Language"),
            gr.Dropdown(langs, label="Target Language")],
    outputs=gr.Textbox(lines=30, label="Translated Text"),
    title="🌍 Language Translation (CPU-friendly)"
)

iface.launch(share=True)
