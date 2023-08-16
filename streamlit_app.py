import gradio as gr
import streamlit as st
from gramformer import Gramformer
import torch

# Create Gramformer instance
gf = Gramformer(models=1, use_gpu=False)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(1212)

# Define function to correct sentences
def correct(sentence):
    res = gf.correct(sentence)  # Gramformer correct
    res = list(res)
    return res[0]  # Return the first value in the res array

# Streamlit app
def streamlit_app():
    st.title('Gramformer Sentence Correction with Gradio and Streamlit')
    st.write("Enter a sentence below to get its grammar-corrected version.")

    # Gradio interface within Streamlit
    app_inputs = gr.inputs.Textbox(lines=2, placeholder="Enter sentence here...")
    interface = gr.Interface(fn=correct,
                            inputs=app_inputs,
                            outputs='text',
                            title='Gramformer Sentence Correction')

    # Streamlit components
    st_gradio_button = st.button("Launch Gramformer Interface")

    if st_gradio_button:
        st.write("Gramformer Interface:")
        interface.launch()

    # Streamlit text input
    st.subheader("Or, use Streamlit only:")
    sentence_input = st.text_area("Enter a sentence:", "")

    # Correction button
    if st.button("Correct"):
        if sentence_input:
            corrected_sentence = correct(sentence_input)
            st.write("Corrected Sentence:")
            st.write(corrected_sentence)

if __name__ == "__main__":
    streamlit_app()
