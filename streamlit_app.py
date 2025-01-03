from supabase import create_client
import fitz  # PyMuPDF for PDF processing
import spacy
import pandas as pd
from io import BytesIO
import json
import streamlit as st
import plotly.graph_objects as go
import requests

# Connect to Supabase
SUPABASE_URL = "https://vbgxuijebobixzrqgvys.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZiZ3h1aWplYm9iaXh6cnFndnlzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzU4ODk1MjIsImV4cCI6MjA1MTQ2NTUyMn0.xchbHvyHL3Y1EQ5SQbKMA--CtVlRXsPNUieXTSRZYPY"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize spaCy
nlp = spacy.load('en_core_web_sm')

# Constants
BUCKET_NAME = "pdf"
FOLDER_PATH = "public"
JSON_FOLDER = "json"

def create_dataframe_from_subfolders(bucket_name, folder_path):
    def list_subfolders(bucket_name, folder_path):
        # List subfolders in a folder (excluding .emptyfolderplaceholder)
        response = supabase.storage.from_(bucket_name).list(path=folder_path)
        subfolders = [item['name'] for item in response if item['name'] != '.emptyfolderplaceholder']
        return subfolders

    def list_files_in_subfolder(bucket_name, folder_path):
        # List all files in a specific subfolder
        response = supabase.storage.from_(bucket_name).list(path=folder_path)
        return [item['name'] for item in response]
    
    # List all subfolders
    subfolders = list_subfolders(bucket_name, folder_path)
    
    # Filter out `.emptyfolderplaceholder` from subfolders
    subfolders = [subfolder for subfolder in subfolders if subfolder != '.emptyfolderplaceholder']
    
    # Create a list to store filenames, keywords, and subfolder names
    data = []
    
    for subfolder in subfolders:
        subfolder_path = f"{folder_path}/{subfolder}" if folder_path else subfolder
        # Get files in the subfolder, excluding `.emptyfolderplaceholder`
        files = [
            file for file in list_files_in_subfolder(bucket_name, subfolder_path) 
            if file != '.emptyfolderplaceholder'
        ]
        
        # Add file names, empty keywords, and the subfolder name to the data list
        for file in files:
            data.append([file, "", subfolder])  # The third column is the subfolder name
    
    # Create the DataFrame with the three columns: 'File Name', 'Keywords', and 'Subfolder'
    df = pd.DataFrame(data, columns=["File Name", "Keywords", "Subfolder"])
    
    return df

# Define the domain keywords and their weights
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

def preprocessing(dataframe, bucket_name, main_folder="public"):
    def process_file_in_memory(bucket_name, file_path):
        try:
            # Download the file from Supabase storage
            response = supabase.storage.from_(bucket_name).download(file_path)
            pdf_content = BytesIO(response)
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            extracted_text = ""
            # Extract text from all pages of the PDF
            for page in doc:
                extracted_text += page.get_text("text")
            return extracted_text
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None

    def clean_text_with_spacy(text):
        doc = nlp(text)
        # Clean the text by removing stopwords and applying lemmatization
        cleaned_text = ' '.join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
        return cleaned_text

    # Initialize an empty dictionary to store processed data by subfolder
    structured_data = {"data-scientist": [], "database-management": [], "web-designing": []}

    # Iterate over each row in the dataframe (each representing a file entry)
    for _, row in dataframe.iterrows():
        file_name = row['File Name']
        subfolder = row['Subfolder']
        
        if pd.notna(file_name) and pd.notna(subfolder):  # Ensure file_name and subfolder are not NaN
            # Construct the correct file path: public/subfolder/file_name
            file_path = f"{main_folder}/{subfolder}/{file_name}"
            
            # Extract and clean the file's text
            extracted_text = process_file_in_memory(bucket_name, file_path)
            if extracted_text:
                cleaned_text = clean_text_with_spacy(extracted_text)
                
                # Match keywords in the extracted text and calculate scores
                matched_keywords = {}
                total_score = 0
                for skill, weight in skills[subfolder].items():
                    if skill.lower() in cleaned_text.lower():
                        matched_keywords[skill] = weight
                        total_score += weight  # Add the weight of the matched keyword
                
                # Append the processed file name, matched keywords, and their weights
                structured_data[subfolder].append({
                    "File Name": file_name,
                    "Keywords": matched_keywords,
                    "Score": int(total_score)  # Convert to integer
                })

    # Function to save structured data to separate JSON files and upload them to Supabase
    def save_data_to_json_and_upload(data, subfolder):
        # Sort the data by Score in descending order
        data_sorted = sorted(data, key=lambda x: x['Score'], reverse=True)

        # Add rank column based on sorted order
        for index, resume in enumerate(data_sorted, start=1):
            resume['Rank'] = index  # Assign rank based on position

        # Save the structured data for a specific subfolder to a local JSON file
        file_name = f"{subfolder}_structured_data.json"
        with open(file_name, 'w') as json_file:
            json.dump(data_sorted, json_file, indent=4)

        # Check if the file already exists in Supabase and delete it if it does
        try:
            existing_files = supabase.storage.from_(bucket_name).list("json")  # List files in the 'json' folder
            # Check if the file already exists
            if file_name in [file['name'] for file in existing_files]:
                print(f"File {file_name} already exists. Deleting and uploading a new file.")
                # Delete the existing file
                supabase.storage.from_(bucket_name).remove([f'json/{file_name}'])
            
            # Upload the new JSON file to Supabase
            with open(file_name, 'rb') as file:
                response = supabase.storage.from_(bucket_name).upload(f'json/{file_name}', file)
            print(f"File {file_name} uploaded successfully!")

        except Exception as e:
            print(f"Error uploading {file_name}: {e}")

    # Remove the 'Keywords' column from each subfolder's data before saving
    for subfolder in structured_data:
        for resume in structured_data[subfolder]:
            del resume['Keywords']  # Drop the 'Keywords' column

        # Save and upload the structured data for each subfolder as a separate JSON file
        save_data_to_json_and_upload(structured_data[subfolder], subfolder)

    return structured_data  # Optionally, return the structured data
  # Optionally, return the structured data

# Define Streamlit UI
# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Intro", "Job Seeker", "Recruiter"])

# Page content based on selection
if page == "Intro":
    st.title("Resume Ranking System")
    st.subheader("Welcome to the Resume Ranking System!")
    st.write("This system helps job seekers upload their resumes and get ranked based on their skills. Recruiters can view the leaderboard and download the resumes.")

elif page == "Job Seeker":
    # Job Seeker Section
    st.header("Job Seeker")
    st.subheader("Upload Your Resume")
    
    # Domain selection and file upload
    domain = st.selectbox("Select Your Domain", ["data-scientist", "database-management", "web-designing"])
    uploaded_file = st.file_uploader("Upload your PDF Resume", type="pdf")
    
    # Button to upload the resume
    if uploaded_file and st.button("Upload Resume"):
        # Define the file path in Supabase
        file_path = f"{FOLDER_PATH}/{domain}/{uploaded_file.name}"
        
        try:
            # Check if the file already exists in Supabase
            existing_files = supabase.storage.from_(BUCKET_NAME).list(f"{FOLDER_PATH}/{domain}")
            
            if any(file['name'] == uploaded_file.name for file in existing_files):
                st.warning(f"The file '{uploaded_file.name}' already exists in the '{domain}' folder.")
                overwrite = st.radio("Do you want to overwrite the existing file?", ("Yes", "No"))
                
                if overwrite == "No":
                    st.info("Upload cancelled. The file was not overwritten.")
                else:
                    # Delete the existing file before re-uploading
                    supabase.storage.from_(BUCKET_NAME).remove([file_path])
                    response = supabase.storage.from_(BUCKET_NAME).upload(file_path, uploaded_file.getvalue())
                    st.success("File overwritten successfully!")
            else:
                # Upload the file as it doesn't exist
                response = supabase.storage.from_(BUCKET_NAME).upload(file_path, uploaded_file.getvalue())
                st.success("Resume uploaded successfully!")
                
        except Exception as e:
            st.error(f"Failed to upload the resume: {e}")
    
    # Button to get the rank of the latest uploaded file
    if uploaded_file and st.button("Get Rank"):
        # Process files and display rank for the latest uploaded file only
        df = create_dataframe_from_subfolders(BUCKET_NAME, FOLDER_PATH)
        structured_data = preprocessing(df, BUCKET_NAME, FOLDER_PATH)
        
        # Get the leaderboard for the selected domain
        leaderboard = structured_data[domain]
        # Find the latest uploaded file in the domain
        latest_file = None
        for file in leaderboard:
            if file['File Name'] == uploaded_file.name:
                latest_file = file
                break
        
        if latest_file:
            st.write(f"Rank for the Latest Uploaded File: {uploaded_file.name}")
            st.write(f"Rank: {latest_file['Rank']}")
        else:
            st.error("Could not find the rank for the uploaded file.")

# To persist the leaderboard and download links across reruns, we use session state
if 'leaderboard_df' not in st.session_state:
    st.session_state['leaderboard_df'] = None

# Recruiter Section
if page == "Recruiter":
    st.title("Resume Ranking System")
    st.header("Recruiter")
    st.subheader("View Leaderboard")
    selected_domain = st.selectbox("Select Domain", ["data-scientist", "database-management", "web-designing"], key="recruiter_domain")

    if st.button("Show Leaderboard"):
        json_file_path = f"{JSON_FOLDER}/{selected_domain}_structured_data.json"
        try:
            # Download the leaderboard JSON from Supabase
            response = supabase.storage.from_(BUCKET_NAME).download(json_file_path)
            leaderboard_data = json.load(BytesIO(response))
            
            # Sort leaderboard by rank
            leaderboard_data_sorted = sorted(leaderboard_data, key=lambda x: x['Rank'])
            leaderboard_df = pd.DataFrame(leaderboard_data_sorted)

            # Store the leaderboard in session state
            st.session_state['leaderboard_df'] = leaderboard_df

            # Add "Download PDF" column with the file URLs for links
            FOLDER_PATH = "pdf"  # Your folder path
            leaderboard_df["Download PDF"] = leaderboard_df["File Name"].apply(
                lambda x: f'{SUPABASE_URL}/storage/v1/object/public/{FOLDER_PATH}/{selected_domain}/{x}'
            )

            # Display leaderboard using Plotly Table (without "Download PDF" column)
            st.subheader(f"Leaderboard for {selected_domain}")
            fig = go.Figure(
                data=[go.Table(
                    header=dict(
                        values=["File Name", "Score", "Rank"],
                        fill_color='rgba(255, 87, 51, 0.5)',
                        align='center',
                        font=dict(size=20),
                        height=35
                    ),
                    cells=dict(
                        values=[leaderboard_df["File Name"], 
                                leaderboard_df["Score"], 
                                leaderboard_df["Rank"]],
                        fill_color='teal',
                        align='center',
                        font=dict(size=18),
                        height=30
                    )
                )]
            )

            # Update layout for better visualization
            fig.update_layout(width=900, height=500)

            # Render the table using Plotly
            st.plotly_chart(fig, use_container_width=True)

            # Manually add clickable download links using st.download_button
            st.subheader("Download Links:")
            for _, row in leaderboard_df.iterrows():
                pdf_url = f"{SUPABASE_URL}/storage/v1/object/public/pdf/public/{selected_domain}/{row['File Name']}"
                st.download_button(
                    label=f"Download {row['File Name']}",
                    data=requests.get(pdf_url).content,
                    file_name=row['File Name'],
                    mime="application/pdf"
                )

            # Trigger snowflake effect
            st.snow()

        except Exception as e:
            st.error(f"Failed to load leaderboard: {e}")

    # If leaderboard already loaded, don't reload it
    elif st.session_state['leaderboard_df'] is not None:
        leaderboard_df = st.session_state['leaderboard_df']
        st.subheader(f"Leaderboard for {selected_domain}")
        fig = go.Figure(
            data=[go.Table(
                header=dict(
                    values=["File Name", "Score", "Rank"],
                    fill_color='rgba(255, 87, 51, 0.5)',
                    align='center',
                    font=dict(size=20),
                    height=35
                ),
                cells=dict(
                    values=[leaderboard_df["File Name"], 
                            leaderboard_df["Score"], 
                            leaderboard_df["Rank"]],
                    fill_color='teal',
                    align='center',
                    font=dict(size=18),
                    height=30
                )
            )]
        )

        # Update layout for better visualization
        fig.update_layout(width=900, height=300)

        # Render the table using Plotly
        st.plotly_chart(fig, use_container_width=True)

        # Manually add clickable download links using st.download_button
        st.subheader("Download Links:")
        for _, row in leaderboard_df.iterrows():
            pdf_url = f"{SUPABASE_URL}/storage/v1/object/public/pdf/public/{selected_domain}/{row['File Name']}"
            st.download_button(
                label=f"Download {row['File Name']}",
                data=requests.get(pdf_url).content,
                file_name=row['File Name'],
                mime="application/pdf"
            )
