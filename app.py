# app.py  — minimal, no user-adjustable knobs
import os
import streamlit as st
from translator import load_translator

st.set_page_config(page_title="CN→EN Translator", page_icon="🌐", layout="centered")
st.title("CN → EN Translator")

# Root folder that contains your model subfolders
RUN_DIR = os.environ.get("RUN_DIR", ".")  # e.g., "." or absolute path

# --- Choose backend only ---
model_type = st.selectbox("Choose Model", ("SMT", "NMT", "Hybrid"), index=1)

# Map UI -> folder (parent "run" dir, not best_model) and backend hint
FOLDER_MAP = {
    "SMT":    "smt/run",     # expects: best_model/ibm1_s2t.json
    "NMT":    "nmt/run",     # expects: best_model/{config.json, weights, tokenizer}
    "Hybrid": "hybrid/run",  # expects: tokenizer/bpe_joint.model, nmt_best.pth, ibm1.pkl
}
PREFER_MAP = {"SMT": "smt", "NMT": "marian", "Hybrid": "hybrid"}

run_root = os.path.join(RUN_DIR, FOLDER_MAP[model_type])
check_path = run_root if model_type == "Hybrid" else os.path.join(run_root, "best_model")

st.caption(f"Selected folder: `{run_root}`")
if os.path.isdir(check_path):
    st.success(f"Found model artifacts in: `{check_path}`")
else:
    st.warning(f"Expected files not found at: `{check_path}`")

# --- Fixed generation settings (no UI) ---
FIXED_NUM_BEAMS = 5
FIXED_MAX_NEW   = 128
FIXED_NGRAM     = 3
FIXED_LEN_PEN   = 1.0

# Device: auto-pick CUDA if available (no UI)
DEVICE = None  # translator will choose CUDA if available

@st.cache_resource(show_spinner=True)
def get_translator_cached(run_dir: str, prefer: str, device):
    return load_translator(run_dir, prefer=prefer, device=device)

# Load translator
translator = None
err = None
try:
    lr = get_translator_cached(run_root, PREFER_MAP[model_type], DEVICE)
    translator = lr.translator
    st.caption(f"Loaded backend: **{lr.backend}** — {lr.info}")
except Exception as e:
    err = e
    st.error(f"Failed to load translator: {e}")

st.markdown("---")
cn_input = st.text_area("Enter Chinese sentence:", height=120, placeholder="输入中文句子…")

if st.button("Translate", use_container_width=True):
    if err is not None:
        st.error("Translator not loaded. Fix the error above.")
    elif not cn_input.strip():
        st.warning("Please enter a Chinese sentence.")
    else:
        with st.spinner("Translating..."):
            try:
                if model_type == "Hybrid":
                    out, info, table = translator.translate(
                    cn_input,
                    num_beams=FIXED_NUM_BEAMS,
                    max_new_tokens=FIXED_MAX_NEW,
                    no_repeat_ngram_size=FIXED_NGRAM,
                    length_penalty=FIXED_LEN_PEN,
                    return_info=True,
                    return_table=True,   # ✅ 新增
                )
                    st.subheader("Translation")
                    st.info(out or "(empty)")
                    st.dataframe(table)
                else:
                    out, info = translator.translate(
                    cn_input,
                    num_beams=FIXED_NUM_BEAMS,
                    max_new_tokens=FIXED_MAX_NEW,
                    no_repeat_ngram_size=FIXED_NGRAM,
                    length_penalty=FIXED_LEN_PEN,
                    return_info=True,
                )
                    st.subheader("Translation")
                    st.info(out or "(empty)")
                
                # st.caption(info)  # uncomment if you want backend/debug info shown
            except Exception as e:
                st.error(f"Translation failed: {e}")
