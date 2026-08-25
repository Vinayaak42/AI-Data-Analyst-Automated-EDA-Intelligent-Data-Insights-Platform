# ============================================================
# LOCAL LLM ENGINE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_local_llm():

    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
        )

    except Exception:
        return None

    MODEL_NAME = "google/flan-t5-base"

    try:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME
        )

        model.to(device)
        model.eval()

        return {
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
        }

    except Exception as e:

        print(
            f"LLM loading failed: {e}"
        )

        return None


def generate_local_text(
    prompt,
    max_tokens=300
):

    llm = load_local_llm()

    if llm is None:
        return None

    try:

        import torch

        tokenizer = llm["tokenizer"]
        model = llm["model"]
        device = llm["device"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=2,
                early_stopping=True,
            )

        text = tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        )

        return text.strip()

    except Exception as e:

        print(
            f"Generation error: {e}"
        )

        return None