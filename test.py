import streamlit as st
from gramformer import Gramformer
import torch
import spacy

# def set_seed(seed):
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# set_seed(1212)

gf = Gramformer(models=1, use_gpu=False)

def main():
    st.title("Sentence Correction App")
    st.write("Enter a sentence, and I'll correct it using Gramformer.")

    input_sentence = st.text_input("Enter a sentence:")

    if input_sentence:
        corrected_sentences = gf.correct(input_sentence, max_candidates=1)
        st.write("**Original Sentence:**", input_sentence)

        st.write("**Corrected Sentences:**")
        for corrected_sentence in corrected_sentences:
            st.write(corrected_sentence)

if __name__ == "__main__":
    main()
