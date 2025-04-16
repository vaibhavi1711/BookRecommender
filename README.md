# Semantic Book Recommender using LLMs

This project is an intelligent, emotion-aware book recommender system. It allows users to search for books using natural language queries, classify them as Fiction or Non-Fiction, and sort them by emotional tone (e.g., joyful, suspenseful, sad, etc). The system is powered by LLMs, semantic search, and a user-friendly Gradio interface.

🔑 Key Features
<br>

<br>
🧹 Text Data Cleaning & Exploration
<br>
Notebook: data-exploration.ipynb

🧠 Semantic Vector Search
<br>
Build a vector database using OpenAI embeddings to retrieve books similar to user queries.
<br>
Notebook: vector-search.ipynb

🏷️ Zero-Shot Text Classification
<br>
Classify books as "Fiction" or "Non-Fiction" using LLMs.
<br>
Notebook: text-classification.ipynb

💬 Emotion-Based Sentiment Analysis
<br>
Extract emotions like joy, fear, sadness, surprise, etc., to sort recommendations by tone.
<br>
Notebook: sentiment-analysis.ipynb

🌐 Interactive Web App with Gradio
<br>
Let users explore and receive book recommendations in an intuitive UI.
<br>
Script: gradio-dashboard.py

🛠 Requirements
<br>
This project requires the following packages:
<br>

-kagglehub
<br>
-pandas
<br>
-matplotlib
<br>
-seaborn
<br>
-python-dotenv
<br>
-langchain-community
<br>
-langchain-opencv
<br>
-langchain-chroma
<br>
-transformers
<br>
-gradio
<br>
-ipywidgets
