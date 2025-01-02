import spacy
import pandas as pd
import fitz  # PyMuPDF
import json
import streamlit as st
from io import BytesIO
import subprocess
import os

# Initialize spaCy model
def download_spacy_model():
    try:
        # Try to load the model
        nlp = spacy.load("en_core_web_sm")
        print("Model loaded successfully.")
        return nlp
    except OSError:
        # If the model isn't found, try to install it
        print("Model not found. Attempting to download...")
        try:
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            nlp = spacy.load("en_core_web_sm")
            print("Model downloaded and loaded successfully.")
            return nlp
        except subprocess.CalledProcessError as e:
            print(f"Error downloading model: {e}")
            return None  # Or handle the error as needed
        except OSError:
            print("Failed to load the model after downloading.")
            return None

nlp = download_spacy_model()
if nlp is None:
    print("Model could not be loaded or downloaded.")
else:
    # Proceed with the rest of the code
    pass

# Define the domain-specific skills
skills = {
    "data-scientist": {"python": 1.3, "tensorflow": 1.5, "machine learning": 1.5, "deep learning": 1.8, "nlp": 2.0, "sql": 1.5},
    "database-management": {"sql": 1.3, "mysql": 1.5, "nosql": 1.8, "mongodb": 1.8, "postgresql": 1.8, "oracle": 1.5},
    "web-designing": {"html": 1.3, "css": 1.3, "javascript": 1.5, "photoshop": 1.5, "bootstrap": 1.5, "ui/ux": 1.5},
}

# Function to process resume in memory
def process_resume(uploaded_file):
    pdf_content = BytesIO(uploaded_file.read())
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    extracted_text = ""
    for page in doc:
        extracted_text += page.get_text("text")
    return extracted_text

# Function to clean and extract keywords from text
def clean_text_with_spacy(text):
    doc = spacy.load("en_core_web_sm")(text)
    cleaned_text = ' '.join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
    return cleaned_text

# Function to rank resumes based on skills
def rank_resume_by_domain(extracted_text, domain):
    cleaned_text = nlp(extracted_text)
    matched_keywords = {}
    total_score = 0
    for skill, weight in skills[domain].items():
        if skill.lower() in cleaned_text.lower():
            matched_keywords[skill] = weight
            total_score += weight
    return matched_keywords, total_score

# Streamlit app UI
def main():
    st.title("Resume Ranking System")

    # Domain Selection
    user_type = st.selectbox("Select User Type", ["Job Seeker", "Recruiter"])

    # Job Seeker End
    if user_type == "Job Seeker":
        st.header("Upload Your Resume")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file:
            domain = st.selectbox("Select Your Domain", ["data-scientist", "database-management", "web-designing"])
            extracted_text = process_resume(uploaded_file)
            matched_keywords, total_score = rank_resume_by_domain(extracted_text, domain)
            st.subheader(f"Your Ranking for {domain.capitalize()} Domain")
            st.write(f"Matched Keywords: {matched_keywords}")
            st.write(f"Total Score: {total_score}")

            # Optionally, add the ranked resume to the leaderboard
            if "leaderboard" not in st.session_state:
                st.session_state.leaderboard = {domain: []}
            st.session_state.leaderboard[domain].append({"File Name": uploaded_file.name, "Score": total_score})

    # Recruiter End
    if user_type == "Recruiter":
        st.header("View Leaderboards")
        domain = st.selectbox("Select Domain", ["data-scientist", "database-management", "web-designing"])
        if "leaderboard" in st.session_state and domain in st.session_state.leaderboard:
            leaderboard_data = sorted(st.session_state.leaderboard[domain], key=lambda x: x["Score"], reverse=True)
            st.subheader(f"Leaderboard for {domain.capitalize()} Domain")
            df = pd.DataFrame(leaderboard_data)
            st.table(df[["File Name", "Score"]])

            # Option to download resumes
            selected_resume = st.selectbox("Select a Resume to Download", df["File Name"].values)
            if selected_resume:
                file = [item for item in leaderboard_data if item["File Name"] == selected_resume][0]
                st.download_button("Download Resume", uploaded_file, file_name=selected_resume)

if __name__ == "__main__":
    main()
