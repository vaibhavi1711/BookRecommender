import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr
from scripts.regsetup import description

# Load environment variables
load_dotenv()

# Prepare books dataset
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Ensure tagged_description.txt exists and is valid
file_path = "tagged_description.txt"

if not os.path.exists(file_path):
    # Create a default file if missing
    with open(file_path, "w", encoding="utf-8") as file:
        file.write("Default content for tagged description.\n")
    print(f"'{file_path}' was not found. A default file has been created.")

from langchain.schema import Document

# Check for valid file content manually
try:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        if not content.strip():
            raise ValueError(f"The file '{file_path}' is empty.")
except Exception as e:
    raise RuntimeError(
        f"Failed to load or read from '{file_path}'. Make sure it exists and contains valid plain text."
    ) from e

# Wrap content manually instead of using TextLoader
raw_documents = [Document(page_content=content)]


# Process documents
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search_with_score(query, k=initial_top_k)
    if not recs:
        raise ValueError("No recommendations found based on the input query.")
    books_list = [int(doc.page_content.strip('"').split()[0]) for doc, _ in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


# Prepare categories and tones
categories = ["All"] + list(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Define Gradio Dashboard
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book",
                                placeholder="e.g., 'A story about forgiveness'")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone", value="All")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommended Books")
    output = gr.Gallery(label="Recommended Books", columns=8, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

if __name__ == "__main__":
    dashboard.launch()
