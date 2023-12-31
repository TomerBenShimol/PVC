import streamlit as st
from functions import *
import db_service

header = st.container()
dataset = st.container()
dataset_statistics = st.container()
filler = st.container()
model = st.container()
text_input_container = st.empty()


with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with header:
    st.markdown(
        '<h1 class="main-header">Classification models on this page were trained on - "SCE dataset"</h1>',
        unsafe_allow_html=True,
    )

with dataset:
    st.markdown(
        '<h3 class="secondary-header">SCE dataset</h3><p class="description">Annotated by <a href="https://www.linkedin.com/in/tomerbenshimol/" class="link1">Tomer Ben Shimol</a> & <a href="https://www.linkedin.com/in/eli-amuyev-224a4b210/" class="link1">Eliyahu Amuyev</a><p>',
        unsafe_allow_html=True,
    )

with dataset_statistics:
    # Import data
    dataset = db_service.get_dataset(1)
    st.write(dataset.head())
    st.markdown(
        f'<p class="description">This dataset contains {dataset.shape[0]} issues, of which {dataset.Classification.value_counts()[0]} are not related to privacy violations and {dataset.Classification.value_counts()[1]} are related to privacy violations. The word average is {avg_words(dataset)} and the character average is {avg_chars(dataset)}<p>',
        unsafe_allow_html=True,
    )

with model:
    st.header("SCE Model")
    model, metrics = db_service.get_model(1)
    metrics = metrics.split("\n")
    for i in range(len(metrics)):
        st.text(metrics[i])
    text = st.text_input(
        "", placeholder="Write some text to classify...", key="text_input_1"
    )
    result = -1
    if len(text) == 1:
        text_input_container.write(
            "There must be at least one word that is at least two characters long"
        )
    if text and len(text) > 1:
        st.write("")
        result = new_prediction(model, 1, text)
        if result == 1:
            text_input_container.write("Privacy related! ⛔️")
        if result == 0:
            text_input_container.write("Non privacy related 🙏🏻")
        st.session_state.input_text_1 = ""
with filler:
    st.text("")
