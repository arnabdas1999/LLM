import streamlit as st
import torch
import tiktoken
from typing import Tuple, Optional
from model_utils import GPTModel, GPT_CONFIG_124M, generate, load_weights_into_gpt
from gpt_download3 import download_and_load_gpt2
import os

# --- Constants ---
PAGE_TITLE = "Pretrained GPT-2"
PAGE_ICON = "🤖"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Page Config ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
st.title(f"{PAGE_ICON} {PAGE_TITLE} (124M)")

# --- Sidebar Controls ---
st.sidebar.header("Generation Parameters")
max_tokens = st.sidebar.slider("Max New Tokens", min_value=10, max_value=200, value=50, help="Maximum number of tokens to generate.")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1, help="Controls randomness: higher is more random.")
top_k = st.sidebar.slider("Top-K", min_value=1, max_value=50, value=10, help="Limits vocabulary to top K tokens.")

# --- Model Loading ---
@st.cache_resource
def load_model() -> Tuple[Optional[GPTModel], torch.device]:
    """
    Downloads and loads the pretrained GPT-2 model (124M) weights from OpenAI.
    
    Returns:
        Tuple[Optional[GPTModel], torch.device]: The loaded model and device, or (None, device) if failed.
    """
    st.info("Loading/Downloading GPT-2 124M weights...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(current, total, desc):
        if total > 0:
            progress = min(current / total, 1.0) # Ensure it doesn't exceed 1.0
            progress_bar.progress(progress)
        status_text.text(f"Processing {desc}: {current}/{total} bytes")
    
    try:
        settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2", progress_callback=update_progress)
        
        # Verify config matches what we expect from download (standard 124M)
        # settings from download_and_load_gpt2 might differ slightly in keys but logic is same.
        # We stick to our GPT_CONFIG_124M structure but use downloaded params.
        
        model = GPTModel(GPT_CONFIG_124M)
        load_weights_into_gpt(model, params)
        model.to(DEVICE)
        model.eval()
        
        status_text.empty()
        progress_bar.empty()
        st.success("Model loaded successfully!")
        return model, DEVICE
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.exception(e) # Show full traceback
        return None, DEVICE

# --- Tokenizer Loading ---
@st.cache_resource
def load_tokenizer():
    """
    Loads the tiktoken tokenizer for GPT-2.
    """
    try:
        return tiktoken.get_encoding("gpt2")
    except Exception as e:
        st.error(f"Failed to load tokenizer: {e}")
        return None

# --- Initialization ---
model, device = load_model()
tokenizer = load_tokenizer()

if model is None or tokenizer is None:
    st.warning("Application halted due to missing resources.")
    st.stop()

# --- Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Enter your prompt here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Prepare input
        try:
            encoded_prompt = tokenizer.encode(prompt, allowed_special={'<|endoftext|>'})
            encoded_tensor = torch.tensor(encoded_prompt).unsqueeze(0).to(device)

            with st.spinner("Generating..."):
                with torch.no_grad():
                    generated_ids = generate(
                        model=model,
                        idx=encoded_tensor,
                        max_new_tokens=max_tokens,
                        context_size=GPT_CONFIG_124M["context_length"],
                        temperature=temperature,
                        top_k=top_k
                    )
            
            # Decode response (remove input prompt from output)
            generated_slice = generated_ids[0, len(encoded_prompt):]
            decoded_response = tokenizer.decode(generated_slice.tolist())
            
            message_placeholder.markdown(decoded_response)
            st.session_state.messages.append({"role": "assistant", "content": decoded_response})
            
        except Exception as e:
            st.error(f"An error occurred during generation: {e}")
