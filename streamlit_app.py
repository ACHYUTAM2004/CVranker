import spacy
import pandas as pd
import fitz  # PyMuPDF
import json
import streamlit as st
from io import BytesIO
import subprocess
import os

# Initialize spaCy model
nlp = spacy.load('en_core_web_sm')

# Define the domain-specific skills
skills = {
    "data-scientist": {"python": 1.3, "tensorflow": 1.5, "machine learning": 1.5, "deep learning": 1.8, "natural language processing": 2.0,
                     "nlp": 2.0, "computer vision": 2.0, "data analysis": 1.5, "data visualization": 1.5, "sql": 1.5, "big data": 1.8,
                     "hadoop": 1.8, "spark": 1.8, "tableau": 1.5, "statistics": 1.5, "probability": 1.5, "data mining": 1.5,
                     "linear regression": 1.5, "logistic regression": 1.5, "decision trees": 1.5, "random forest": 1.5,
                     "support vector machines": 1.5, "neural networks": 1.8, "clustering": 1.5, "classification": 1.5,
                     "regression": 1.5, "feature engineering": 1.5, "model evaluation": 1.5, "model selection": 1.5,
                     "ensemble methods": 1.5, "dimensionality reduction": 1.5, "unsupervised learning": 1.5,
                     "supervised learning": 1.5, "reinforcement learning": 1.8, "time series analysis": 1.5, "anomaly detection": 1.5,
                     "association rule mining": 1.5, "collaborative filtering": 1.5, "content-based filtering": 1.5,
                     "recommendation systems": 1.5, "sentiment analysis": 1.8, "image processing": 1.5, "object detection": 1.5,
                     "image segmentation": 1.5, "image classification": 1.5, "image recognition": 1.5, "convolutional neural networks": 1.8,
                     "recurrent neural networks": 1.8, "generative adversarial networks": 1.8, "transformers": 1.8, "bert": 1.8,
                     "gpt-3": 1.8, "xgboost": 1.5, "lightgbm": 1.5, "catboost": 1.5, "scikit-learn": 1.3, "pandas": 1.5, "numpy": 1.5,
                     "matplotlib": 1.3, "seaborn": 1.3, "plotly": 1.5, "keras": 1.8, "pytorch": 1.8, "power bi": 1.5, "qlik": 1.7, "sas": 1.7, "mysql": 1.5},

    "database-management": {"sql": 1.3, "mysql": 1.5, "database optimization": 1.5, "nosql": 1.8, "mongodb": 1.8, "postgresql": 1.8, "sql server": 1.5,
                 "oracle": 1.5, "database design": 1.5, "data modeling": 1.5, "data warehousing": 1.8, "etl": 1.8, "database administration": 1.5,
                 "database security": 1.8, "stored procedures": 1.5, "triggers": 1.5, "views": 1.5, "indexes": 1.5, "database tuning": 1.8,
                 "database management": 1.8, "database development": 1.8, "database migration": 1.8, "database backup": 1.8, "database recovery": 1.8,
                 "database replication": 1.5, "database clustering": 1.5, "database sharding": 1.5, "database monitoring": 1.5, "database performance": 1.5,
                 "database scalability": 1.3, "database reliability": 1.3, "database availability": 1.3},

    "web-designing": {"html": 1.3, "css": 1.3, "javascript": 1.5, "photoshop": 1.5, "web design": 1.5,
                     "ui/ux": 1.5, "responsive design": 1.5, "bootstrap": 1.5, "jquery": 1.8, "adobe xd": 1.5,
                     "figma": 1.8, "sketch": 1.5, "wireframing": 1.5, "prototyping": 1.5, "usability": 1.5,
                     "user experience": 1.5, "user interface": 1.5, "color theory": 1.5, "typography": 1.5,
                     "grid systems": 1.5, "layout": 1.5, "mockups": 1.5, "web development": 1.8, "front-end": 1.8,
                     "back-end": 1.5, "full-stack": 1.5, "seo": 1.5, "google analytics": 1.5, "web performance": 1.5,
                     "web optimization": 1.5, "cross-browser compatibility": 1.5, "web security": 1.5, "web accessibility": 1.5,
                     "web standards": 1.5, "web testing": 1.5, "web maintenance": 1.5, "web hosting": 1.5, "domain management": 1.5,
                     "dns management": 1.5, "web servers": 1.5, "web deployment": 1.5, "version control": 1.5, "git": 1.5,
                     "github": 1.5, "bitbucket": 1.5, "web collaboration": 1.5, "web project management": 1.5, "agile": 1.5,
                     "scrum": 1.5, "kanban": 1.5, "web communication": 1.5, "web consulting": 1.5, "web training": 1.5,
                     "react js": 1.8, "angular js": 1.8, "vue js": 1.8, "node js": 1.8, "express js": 1.8, "sass": 1.5, "less": 1.5,
                     "rust": 2.0, "webassembly": 1.8, "webgl": 1.8, "three.js": 2.0, "d3.js": 2.0, "websockets": 2.0, "graphql": 2.0, "restful": 2.0,
                     "react": 1.8, "angular": 1.8, "vue": 1.8, "node": 1.8, "express": 1.8, "html5": 1.3, "css3": 1.3}
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
def job_seeker_section():
    # Show domain selection first
    st.header("Job Seeker Section")

    domain = st.selectbox("Select Your Domain", ["Data Science", "Web Designing", "Java Development", "Python Development", "Databases"])

    # After selecting domain, show file uploader for PDF
    uploaded_file = None
    if domain:
        st.subheader(f"Upload Your Resume for {domain} domain:")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # If a file is uploaded, proceed with processing
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        # You can process the resume here (e.g., extract text, rank, etc.)
        # Placeholder for resume processing
        st.text("Processing the resume...")
        # After processing, display the leaderboard for the selected domain
        # You would display leaderboard here based on domain selection
        st.text(f"Displaying leaderboard for {domain}...")
    else:
        st.warning("Please upload your resume after selecting a domain.")

def recruiter_section():
    # Show recruiter section where they can view the leaderboard and download resumes
    st.header("Recruiter Section")
    domain = st.selectbox("Select Domain to View Leaderboard", ["Data Science", "Web Designing", "Java Development", "Python Development", "Databases"])

    # Placeholder for leaderboard display (replace with actual leaderboard)
    st.text(f"Displaying leaderboard for {domain} domain...")
    st.text("Download resumes from the leaderboard...")

def main():
    # Main app with options for Job Seeker or Recruiter
    st.title("Resume Ranking System")

    user_type = st.radio("Select User Type", ["Job Seeker", "Recruiter"])

    if user_type == "Job Seeker":
        job_seeker_section()
    else:
        recruiter_section()

if __name__ == "__main__":
    main()

