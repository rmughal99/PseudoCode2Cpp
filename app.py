import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load trained model and tokenizer
@st.cache_resource()
def load_model():
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.load_state_dict(torch.load("pseudo2cpp.h5", map_location=torch.device("cpu")))
    model.eval()
    return model

@st.cache_resource()
def load_tokenizer():
    return T5Tokenizer.from_pretrained("tokenizer")

model = load_model()
tokenizer = load_tokenizer()

# Streamlit UI
st.title("PseudoCode to C++ Converter")
st.markdown("### Enter your pseudocode below:")

user_input = st.text_area("Pseudocode:", "Loop through array and print each element.")

if st.button("Convert to C++"):  
    if user_input.strip():
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=256)
        outputs = model.generate(**inputs)
        cpp_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        st.markdown("### Generated C++ Code:")
        st.code(cpp_code, language="cpp")
    else:
        st.warning("Please enter some pseudocode!")
