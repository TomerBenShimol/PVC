import streamlit as st
from functions import *

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
        '<h1 class="main-header">Classification models on this page were trained on - "Haifa dataset"</h1>',
        unsafe_allow_html=True,
    )

with dataset:
    st.markdown(
        '<h3 class="secondary-header">Hafia dataset</h3><p class="description">Collected by researchers from University of Haifa<p>',
        unsafe_allow_html=True,
    )

with dataset_statistics:
    # Import data
    dataset = db_service.get_dataset(2)
    st.write(dataset.head())
    st.markdown(
        f'<p class="description">This dataset contains {dataset.shape[0]} issues, of which {dataset.Classification.value_counts()[0]} are not related to privacy violations and {dataset.Classification.value_counts()[1]} are related to privacy violations. The word average is {avg_words(dataset)} and the character average is {avg_chars(dataset)}<p>',
        unsafe_allow_html=True,
    )

with model:
    st.header("Haifa Model")
    model, metrics = db_service.get_model(2)
    st.text(metrics)
    text = st.text_input(
        "", placeholder="Write some text to classify...", key="text_input_2"
    )
    result = -1
    if len(text) == 1:
        text_input_container.write(
            "There must be at least one word that is at least two characters long"
        )
    if text and len(text) > 1:
        st.write("")
        result = new_prediction(model, 2, text)
        if result == 1:
            text_input_container.write("Privacy related! ⛔️")
        if result == 0:
            text_input_container.write("Non privacy related 🙏🏻")
        st.session_state.input_text_2 = ""

with filler:
    st.text("")
