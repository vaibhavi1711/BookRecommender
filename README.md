# Semantic Book Recommender using LLMs

This project is an intelligent, emotion-aware book recommender system. It allows users to search for books using natural language queries, classify them as Fiction or Non-Fiction, and sort them by emotional tone (e.g., joyful, suspenseful, sad, etc). The system is powered by LLMs, semantic search, and a user-friendly Gradio interface.

🔑 Key Features
🧹 Text Data Cleaning & Exploration
Notebook: data-exploration.ipynb

🧠 Semantic Vector Search
Build a vector database using OpenAI embeddings to retrieve books similar to user queries.
Notebook: vector-search.ipynb

🏷️ Zero-Shot Text Classification
Classify books as "Fiction" or "Non-Fiction" using LLMs.
Notebook: text-classification.ipynb

💬 Emotion-Based Sentiment Analysis
Extract emotions like joy, fear, sadness, surprise, etc., to sort recommendations by tone.
Notebook: sentiment-analysis.ipynb

🌐 Interactive Web App with Gradio
Let users explore and receive book recommendations in an intuitive UI.
Script: gradio-dashboard.py

🛠 Requirements
This project requires the following packages:

kagglehub
pandas
matplotlib
seaborn
python-dotenv
langchain-community
langchain-opencv
langchain-chroma
transformers
gradio
ipywidgets
