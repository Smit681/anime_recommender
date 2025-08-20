# 🎥 Anime Recommendation Web App  

👉 **Live App:** https://animerecommend.streamlit.app/

---

## 📌 Project Overview
This project is a **personalized anime recommendation system** built as part of my Computer Science Capstone.  
The application helps anime fans discover new titles that match their preferences and mood.  

Unlike static top-anime lists, this tool combines **objective data (ratings, genres, studios, years)** with **subjective mood mapping** (e.g., excited, sad, motivated) to create a truly personal watchlist.  

---

## ✨ Features
- **Input Options**
  - Select favorite anime titles
  - Filter by **genre, year range, studio, rating**
  - Choose your current **mood** (e.g., sad, excited, motivated, relaxed)  

- **Smart Recommendations**
  - Powered by **Machine Learning (ML)**:
    - **K-Means Clustering** groups anime into natural categories (e.g., “fantasy epics,” “slice-of-life comedies”)
    - **K-Nearest Neighbors (KNN) / Cosine Similarity** predicts new titles based on user favorites
  - **Mood-aware suggestions** via sentiment mapping  

- **Visual Insights**
  - Genre distribution of recommended anime  
  - Rating trends over years  
  - Popularity vs. score scatter plot  

- **Friendly UI**
  - Built entirely in **Python with Streamlit**
  - Simple, interactive interface
  - Image + description for each recommendation  

---

## 🛠️ Tech Stack
- **Frontend & Backend**: [Streamlit](https://streamlit.io/)  
- **Machine Learning**: scikit-learn (KMeans, cosine similarity / KNN)  
- **Data Processing**: pandas, numpy  
- **Visuals**: matplotlib & Streamlit native charts  
- **Dataset**: [Anime Data Set for ML – Kaggle](https://www.kaggle.com/datasets/wiltheman/anime-data-set-for-ml)  

---



## 🚀 How to Run Locally
1. Clone this repo:
   git clone https://github.com/<your-username>/anime-recs.git
   cd anime-recs
2. Install dependencies: pip install -r requirements.txt
3. Run the App: streamlit run streamlit_app.py

## 📖 Methods Summary

Descriptive: K-Means Clustering to group anime into natural categories.
Predictive: KNN / cosine similarity to recommend unseen anime based on user favorites.
Prescriptive: Mood → genre mapping adjusts recommendations to fit emotional context.

   

