import os
import pickle
import re
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCT_FILE = os.path.join(BASE_DIR, "data", "processed", "products.csv")
INDEX_FILE = os.path.join(BASE_DIR, "embeddings", "product_faiss.index")
METADATA_FILE = os.path.join(BASE_DIR, "embeddings", "product_metadata.pkl")
VECTORIZER_FILE = os.path.join(BASE_DIR, "embeddings", "product_tfidf.pkl")

load_dotenv(os.path.join(BASE_DIR, ".env"))
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

print("Loading FAISS product knowledge base...")
index = faiss.read_index(INDEX_FILE)
with open(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)
with open(VECTORIZER_FILE, "rb") as f:
    vectorizer = pickle.load(f)
print("FAISS index loaded.")
print("Products in index:", index.ntotal)

products = pd.read_csv(PRODUCT_FILE)
products["product_category"] = products["product_category"].fillna("").astype(str)
products["unit_price"] = pd.to_numeric(products["unit_price"], errors="coerce").fillna(0)
products["rating"] = pd.to_numeric(products["rating"], errors="coerce").fillna(0)

def make_source(product):
    text = (
        f"Product ID: {int(product['product_id'])}\n"
        f"Category: {product['product_category']}\n"
        f"Price: ₹{float(product['unit_price']):.2f}\n"
        f"Rating: {float(product['rating']):.1f}"
    )
    return {
        "product_id": int(product["product_id"]),
        "score": float(product.get("similarity", 0.0)),
        "text": text
    }

def faiss_search(query, top_k=5):
    query_vector = vectorizer.transform([query]).astype(np.float32).toarray()
    faiss.normalize_L2(query_vector)
    scores, indices = index.search(query_vector, top_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        product = metadata.iloc[idx]
        results.append({
            "product_id": int(product["product_id"]),
            "product_category": str(product["product_category"]),
            "unit_price": float(product["unit_price"]),
            "rating": float(product["rating"]),
            "similarity": float(score)
        })
    return results

def search_products(query, top_k=5):
    query_lower = query.lower()
    price_match = re.search(
        r"(?:under|below|less than|within)\s*(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)",
        query_lower
    )
    if price_match:
        max_price = float(price_match.group(1))
        filtered = products[products["unit_price"] <= max_price].copy()
        if len(filtered) > 0:
            filtered = filtered.sort_values(
                by=["rating", "unit_price"],
                ascending=[False, True]
            )
            results = []
            for _, product in filtered.head(top_k).iterrows():
                results.append({
                    "product_id": int(product["product_id"]),
                    "product_category": str(product["product_category"]),
                    "unit_price": float(product["unit_price"]),
                    "rating": float(product["rating"]),
                    "similarity": 1.0
                })
            return results

    rating_words = [
        "high rating",
        "highly rated",
        "best rated",
        "good rating",
        "top rated",
        "highest rated"
    ]
    if any(word in query_lower for word in rating_words):
        filtered = products.sort_values(
            by=["rating", "unit_price"],
            ascending=[False, True]
        )
        results = []
        for _, product in filtered.head(top_k).iterrows():
            results.append({
                "product_id": int(product["product_id"]),
                "product_category": str(product["product_category"]),
                "unit_price": float(product["unit_price"]),
                "rating": float(product["rating"]),
                "similarity": 1.0
            })
        return results

    results = faiss_search(query, top_k=top_k)
    if not results or all(result["similarity"] <= 0 for result in results):
        fallback = products.sort_values(
            by=["rating", "unit_price"],
            ascending=[False, True]
        )
        results = []
        for _, product in fallback.head(top_k).iterrows():
            results.append({
                "product_id": int(product["product_id"]),
                "product_category": str(product["product_category"]),
                "unit_price": float(product["unit_price"]),
                "rating": float(product["rating"]),
                "similarity": 0.0
            })
    return results

def build_context(products_found):
    context = ""
    for product in products_found:
        context += (
            f"Product ID: {product['product_id']}\n"
            f"Category: {product['product_category']}\n"
            f"Price: ₹{product['unit_price']:.2f}\n"
            f"Rating: {product['rating']:.1f}\n"
            f"Similarity Score: {product['similarity']:.4f}\n\n"
        )
    return context

def ask_shopping_assistant(question):
    products_found = search_products(question, top_k=5)
    context = build_context(products_found)
    prompt = f"""You are an AI shopping assistant for an Indian e-commerce platform.
Use ONLY the product information provided below.

PRODUCT INFORMATION:
{context}

USER QUESTION:
{question}

Instructions:
1. Recommend products only from the provided information.
2. Do not invent product names or specifications.
3. Mention Product ID when recommending a product.
4. Mention price and rating when useful.
5. If the available information is insufficient, clearly say that.
6. Keep the answer concise and helpful."""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful and factual e-commerce shopping assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )
        answer = response.choices[0].message.content
        if not answer:
            answer = "I found products, but the AI model did not return a text answer."
    except Exception as e:
        raise RuntimeError(f"OpenRouter API error: {e}") from e

    sources = [make_source(product) for product in products_found]
    return {
        "answer": answer,
        "sources": sources,
        "products": products_found
    }

if __name__ == "__main__":
    print("=" * 60)
    print("SMART RAG SHOPPING ASSISTANT")
    print("=" * 60)
    while True:
        question = input("\nEnter your shopping question (type exit to stop): ")
        if question.lower().strip() == "exit":
            break
        print("\nSearching products...")
        try:
            result = ask_shopping_assistant(question)
            products_found = result["products"]
            answer = result["answer"]
            print("\n" + "=" * 60)
            print("RETRIEVED PRODUCTS")
            print("=" * 60)
            for product in products_found:
                print(
                    f"Product ID: {product['product_id']} | "
                    f"Category: {product['product_category']} | "
                    f"Price: ₹{product['unit_price']:.2f} | "
                    f"Rating: {product['rating']:.1f} | "
                    f"Score: {product['similarity']:.4f}"
                )
            print("\n" + "=" * 60)
            print("AI SHOPPING ASSISTANT")
            print("=" * 60)
            print(answer)
        except Exception as e:
            print(f"\nError: {e}")
    print("\nRAG TEST COMPLETED.")
