import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import torch

# Load model
@st.cache_resource
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')  # small + fast model
    return model

bert_model = load_model()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to compute similarity
def compute_similarity(cv_embedding, jd_embedding):
    return np.dot(cv_embedding, jd_embedding) / (np.linalg.norm(cv_embedding) * np.linalg.norm(jd_embedding))

# Streamlit UI
st.title("📄 CV Selector App")
st.write("Upload a Job Description and multiple CVs. We'll rank them based on matching score!")

# Upload JD
jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

# Upload multiple CVs
cv_files = st.file_uploader("Upload CVs (PDFs)", type=["pdf"], accept_multiple_files=True)

if jd_file and cv_files:
    with st.spinner('Processing...'):
        # Extract JD text and embed
        jd_text = extract_text_from_pdf(jd_file)
        jd_embedding = bert_model.encode(jd_text)

        results = []

        for cv in cv_files:
            cv_text = extract_text_from_pdf(cv)
            cv_embedding = bert_model.encode(cv_text)

            score = compute_similarity(cv_embedding, jd_embedding)
            results.append((cv.name, score))

        # Sort results
        results.sort(key=lambda x: x[1], reverse=True)

        st.subheader("📋 Ranking of CVs (High to Low match):")
        df = pd.DataFrame(results, columns=["CV Name", "Matching Score"])
        st.dataframe(df)

        top_candidate = results[0][0]
        st.success(f"🏆 Best Match: **{top_candidate}**")

