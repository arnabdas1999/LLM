import streamlit as st
import torch
import tiktoken
from typing import Tuple, Optional
from model_utils import GPTModel, GPT_CONFIG_355M, generate
import os

# --- Constants ---
PAGE_TITLE = "Fine-Tuned LLM"
PAGE_ICON = "🧠"
MODEL_PATH = "gpt2-medium355M-sft.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Page Config ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
st.title(f"{PAGE_ICON} Instruction Fine-Tuned LLM (355M)")

# --- Sidebar Controls ---
st.sidebar.header("Generation Parameters")
max_tokens = st.sidebar.slider("Max New Tokens", min_value=10, max_value=200, value=100, help="Maximum number of tokens to generate.")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.5, step=0.1, help="Controls randomness.")
top_k = st.sidebar.slider("Top-K", min_value=1, max_value=50, value=5, help="Limits vocabulary to top K tokens.")

# --- Model Loading ---
@st.cache_resource
def load_model() -> Tuple[Optional[GPTModel], torch.device]:
    """
    Loads the fine-tuned GPT-2 model and weights.
    
    Returns:
        Tuple[Optional[GPTModel], torch.device]: The loaded model and device, or (None, device) if failed.
    """
    model = GPTModel(GPT_CONFIG_355M)
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: `{MODEL_PATH}`. Please ensure it is in the root directory.")
        return None, DEVICE

    try:
        # Load state dict
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Failed to load weights from `{MODEL_PATH}`: {e}")
        st.exception(e)
        return None, DEVICE
        
    model.to(DEVICE)
    model.eval()
    return model, DEVICE

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

# --- Helpers ---
def format_instruction(instruction: str) -> str:
    """
    Formats the user instruction into the template used during training.
    """
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )

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
if prompt := st.chat_input("Enter instruction..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Format input with instruction template
        formatted_prompt = format_instruction(prompt)
        
        try:
            encoded_prompt = tokenizer.encode(formatted_prompt, allowed_special={'<|endoftext|>'})
            encoded_tensor = torch.tensor(encoded_prompt).unsqueeze(0).to(device)

            with st.spinner("Generating response..."):
                with torch.no_grad():
                    generated_ids = generate(
                        model=model,
                        idx=encoded_tensor,
                        max_new_tokens=max_tokens,
                        context_size=GPT_CONFIG_355M["context_length"],
                        temperature=temperature,
                        top_k=top_k
                    )
            
            # Decode and extract response
            # We only want the *new* tokens, which come after the prompt
            generated_slice = generated_ids[0, len(encoded_prompt):]
            decoded_response = tokenizer.decode(generated_slice.tolist())
            
            # Display response
            message_placeholder.markdown(decoded_response)
            st.session_state.messages.append({"role": "assistant", "content": decoded_response})
            
        except Exception as e:
            st.error(f"An error occurred during generation: {e}")
