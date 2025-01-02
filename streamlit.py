import streamlit as st
import pandas as pd
from supabase import create_client
from io import BytesIO
import fitz  # PyMuPDF for PDF processing
import spacy

# Initialize spaCy
nlp = spacy.load('en_core_web_sm')

# Supabase credentials
SUPABASE_URL = "https://oewyazfmpcfoxwjunwpp.supabase.co/"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9ld3lhemZtcGNmb3h3anVud3BwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzU0MDE0NzgsImV4cCI6MjA1MDk3NzQ3OH0.wpJQbLcwvnTO-BW3D4d9R1LrLlUiBONPlzUtUU3Qb8w"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Define job roles and their required skills
JOB_SKILLS = {
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

# Helper function to extract and clean text from a PDF
def extract_text_from_pdf(uploaded_file):
    file_bytes = BytesIO(uploaded_file.read())
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Function to calculate skill match score
def calculate_score(text, skills):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    score = sum(1 for skill in skills if skill.lower() in tokens)
    return score

# Streamlit App
st.title("Resume Ranking System")
st.markdown("""Upload resumes to process and rank them for various job domains.""")

# File uploader widget
uploaded_files = st.file_uploader("Upload Resumes (PDF format)", accept_multiple_files=True, type=["pdf"])
selected_role = st.selectbox("Select Job Role", options=list(JOB_SKILLS.keys()))

if st.button("Process Resumes"):
    if uploaded_files and selected_role:
        st.write("Processing resumes...")
        results = []
        required_skills = JOB_SKILLS[selected_role]

        for uploaded_file in uploaded_files:
            # Extract text from PDF
            text = extract_text_from_pdf(uploaded_file)

            # Calculate skill match score
            score = calculate_score(text, required_skills)

            # Store results
            results.append({
                "File Name": uploaded_file.name,
                "Score": score
            })

        # Convert results to DataFrame
        results_df = pd.DataFrame(results).sort_values(by="Score", ascending=False)

        # Display results
        st.write(f"### Ranking for {selected_role}:")
        st.dataframe(results_df)
    else:
        st.error("Please upload at least one PDF file and select a job role.")

# Option to upload ranked resumes to Supabase
if st.button("Upload Rankings to Supabase"):
    if 'results_df' in locals():
        # Convert DataFrame to list of dictionaries for uploading
        for _, row in results_df.iterrows():
            data = {
                "file_name": row["File Name"],
                "score": row["Score"],
                "job_role": selected_role
            }
            supabase.table("resume_rankings").insert(data).execute()

        st.success("Rankings uploaded successfully!")
    else:
        st.error("No results to upload.")
