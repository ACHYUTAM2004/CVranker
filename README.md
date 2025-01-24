### **Resume Ranking System** 📄  
- **Overview:**  
  A system designed to automatically categorize and rank resumes by specific domains (e.g., Data Science, Web Design, Java Development). It helps recruiters and hiring managers streamline the hiring process by identifying the best candidates based on required skills.  

- **Key Features:**  
  - Categorizes resumes into domains like **Data Science**, **Web Design**, **Database Management**, and more using keyword matching and weighted scoring.  
  - Ranks resumes within each category based on the relevance of skills, projects, and experience.  
  - Provides an intuitive, user-friendly web interface for uploading resumes and viewing results.
  - It has 2 sections job seeker and recruiter.
  - In the job seeker section, user can upload his resume and get the rank of latest uploaded resume.
  - In the recruiter section, a recruiter can see the leaderboard of any domain and can also download any resume with the available download link.

- **Tech Stack:**  
  - **Backend:** Python, Resume Parsing with libraries like **PyPDF2** and **spaCy**   
  - **Storage:** **Supabase** bucket to store resumes in organized folders.  
  - **Deployment:** Deployed on **Streamlit** for easy accessibility.  

- **Challenges Solved:**  
  - Efficiently extracting and analyzing data from various resume formats (PDF, Docx).  
  - Implemented a scoring algorithm to weigh skills, certifications, and projects dynamically.  
  - Ensured domain-specific customization by allowing flexibility in skill weighting.  

- **Future Improvements:**  
  - Incorporate NLP-based semantic analysis to enhance keyword matching and ranking accuracy.  
  - Add support for generating domain-specific feedback for candidates.  
  - Enable multi-language support for international resumes.  

- **Impact:**  
  Significantly reduces the time and effort required in the initial screening of candidates, helping organizations focus on top talent quickly and efficiently.  

- **Live Demo:**  
  The project is live on **Streamlt** and ready for recruiters to test and use.
  
- **Job Seeker Section**
  ![image](https://github.com/user-attachments/assets/b56ddbf2-d1f4-451a-8c19-3dd860e28ad7)

- **Recruiter Section**
  ![image](https://github.com/user-attachments/assets/75702098-d3b2-451f-b2a3-74b58b9abbf8)


