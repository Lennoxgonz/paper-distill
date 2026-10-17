import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_ID = "sshleifer/distilbart-cnn-12-6"
# Encoder context of distilbart-cnn-12-6. Longer abstracts are truncated.
MAX_INPUT_TOKENS = 1024


@st.cache_resource(show_spinner="Loading summarizer model (first use may download ~120MB)…")
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


def summarize_text(text: str, *, max_length: int, min_length: int) -> str:
    """Summarize text with the local DistilBART model.

    Transformers 5 removed the summarization pipeline. Calling the seq2seq
    model directly keeps the same model and the same token-length controls
    the abstract page used to pass to that pipeline.
    """
    tokenizer, model = load_summarizer()
    max_length = max(int(max_length), 8)
    min_length = max(int(min_length), 0)
    if min_length >= max_length:
        min_length = max_length - 1

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    output_ids = model.generate(
        **inputs,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
