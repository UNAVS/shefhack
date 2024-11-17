import streamlit as st
import pandas as pd
import subprocess
import sys

# Install necessary packages in case they're missing
# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

pipenv install keybert
pipenv install keyphrase_vectorizers

from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer

page_bg_img = '''
<style>
body {
background-image: url("https://ibb.co/FhDkBJK");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Initialize KeyBERT model and vectorizer
@st.cache_resource
def load_keybert():
    embedding = 'all-mpnet-base-v2'
    key_model = KeyBERT(model=embedding)
    vectorizer = KeyphraseCountVectorizer(
        spacy_pipeline='en_core_web_sm',
        pos_pattern='<J.*>*<N.*>+',
        stop_words='english',
        lowercase=True
    )
    return key_model, vectorizer

key_model, vectorizer_params = load_keybert()

# Define function to extract keywords
def get_keywords(course_name, course_desc):
    keywords_list = []
    course_name, course_desc = course_name.strip().lower(), course_desc.strip().lower()
    data = course_name + ". " + course_desc
    keywords = key_model.extract_keywords(
        data, vectorizer=vectorizer_params, stop_words='english', top_n=7, use_mmr=True
    )
    keywords_list = list(dict(keywords).keys())
    return ", ".join(keywords_list)

def get_keywords_file(uploaded_file):
    input_data = pd.read_csv(uploaded_file)
    keywords_data = []
    for i in range(len(input_data)):
        curr_keywords = get_keywords(
            input_data["Course Name"][i], input_data["Course Desc"][i]
        )
        keywords_data.append(curr_keywords)
    input_data["Relevant Tags"] = keywords_data

    output_file = "CourseTagGen_Result.csv"
    input_data.to_csv(output_file, index=False)
    return output_file, input_data

# Streamlit App Layout
st.title("College Course Tags Generator")

tab1, tab2 = st.tabs(["Generate From Text Input", "Generate From CSV/XLSX"])

# Tab 1: Generate tags from text input
with tab1:
    st.header("Generate Tags from Text Input")
    course_name = st.text_input("Course Name", placeholder="Enter course name...")
    course_desc = st.text_area("Course Description", placeholder="Enter course description...")
    if st.button("Generate Tags"):
        if course_name and course_desc:
            tags = get_keywords(course_name, course_desc)
            st.success("Relevant Tags:")
            st.write(tags)
        else:
            st.warning("Please fill in both the course name and description.")

# Tab 2: Generate tags from uploaded CSV file
with tab2:
    st.header("Generate Tags from CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file and st.button("Process File"):
        output_file, processed_data = get_keywords_file(uploaded_file)
        st.success("File processed successfully! Download the result below.")
        st.download_button(
            label="Download Result CSV",
            data=processed_data.to_csv(index=False),
            file_name="CourseTagGen_Result.csv",
            mime="text/csv"
        )

