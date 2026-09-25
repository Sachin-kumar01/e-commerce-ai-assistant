import os
import pickle
import numpy as np
import pandas as pd
import faiss

from sklearn.feature_extraction.text import TfidfVectorizer


# =========================================================
# PATHS
# =========================================================

PRODUCT_FILE = "data/processed/products.csv"

INDEX_DIR = "embeddings"

FAISS_INDEX_FILE = os.path.join(
    INDEX_DIR,
    "product_faiss.index"
)

PRODUCT_DATA_FILE = os.path.join(
    INDEX_DIR,
    "product_metadata.pkl"
)

VECTORIZER_FILE = os.path.join(
    INDEX_DIR,
    "product_tfidf.pkl"
)


# =========================================================
# CREATE EMBEDDING DIRECTORY
# =========================================================

os.makedirs(INDEX_DIR, exist_ok=True)


# =========================================================
# LOAD PRODUCTS
# =========================================================

print("=" * 60)
print("FAISS PRODUCT KNOWLEDGE BASE")
print("=" * 60)

print("\nLoading products...")

products = pd.read_csv(PRODUCT_FILE)

print("Products loaded:", len(products))


# =========================================================
# CLEAN DATA
# =========================================================

products = products.copy()

products["product_category"] = (
    products["product_category"]
    .fillna("")
    .astype(str)
)

products["unit_price"] = (
    products["unit_price"]
    .fillna(0)
    .astype(str)
)

products["rating"] = (
    products["rating"]
    .fillna(0)
    .astype(str)
)


# =========================================================
# CREATE PRODUCT TEXT
# =========================================================

products["product_text"] = (
    "Product ID: "
    + products["product_id"].astype(str)
    + ". Category: "
    + products["product_category"]
    + ". Price: "
    + products["unit_price"]
    + ". Rating: "
    + products["rating"]
)


print("\nExample product text:")
print(products["product_text"].iloc[0])


# =========================================================
# TF-IDF VECTORIZATION
# =========================================================

print("\nCreating TF-IDF vectors...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=5000
)

vectors = vectorizer.fit_transform(
    products["product_text"]
)

print("TF-IDF shape:", vectors.shape)


# =========================================================
# CONVERT TO NUMPY
# =========================================================

vectors = vectors.astype(
    np.float32
).toarray()

print("Vector shape:", vectors.shape)


# =========================================================
# NORMALIZE VECTORS
# =========================================================

faiss.normalize_L2(vectors)


# =========================================================
# CREATE FAISS INDEX
# =========================================================

dimension = vectors.shape[1]

print("\nCreating FAISS index...")
print("Vector dimension:", dimension)

index = faiss.IndexFlatIP(dimension)

index.add(vectors)

print("Vectors stored in FAISS:", index.ntotal)


# =========================================================
# SAVE FAISS INDEX
# =========================================================

faiss.write_index(
    index,
    FAISS_INDEX_FILE
)

print("\nFAISS index saved:")
print(FAISS_INDEX_FILE)


# =========================================================
# SAVE PRODUCT METADATA
# =========================================================

metadata = products[
    [
        "product_id",
        "product_category",
        "unit_price",
        "rating"
    ]
].copy()

with open(
    PRODUCT_DATA_FILE,
    "wb"
) as f:

    pickle.dump(
        metadata,
        f
    )


# =========================================================
# SAVE TF-IDF VECTORIZER
# =========================================================

with open(
    VECTORIZER_FILE,
    "wb"
) as f:

    pickle.dump(
        vectorizer,
        f
    )


print("Product metadata saved:")
print(PRODUCT_DATA_FILE)

print("TF-IDF vectorizer saved:")
print(VECTORIZER_FILE)


# =========================================================
# TEST SEARCH
# =========================================================

print("\n" + "=" * 60)
print("TESTING PRODUCT SEARCH")
print("=" * 60)


query = "electronics mobile smartphone"

print("\nSearch Query:")
print(query)


query_vector = vectorizer.transform(
    [query]
).astype(
    np.float32
).toarray()


faiss.normalize_L2(
    query_vector
)


scores, indices = index.search(
    query_vector,
    5
)


print("\nTop 5 Similar Products:")

for rank, (idx, score) in enumerate(
    zip(indices[0], scores[0]),
    start=1
):

    product = metadata.iloc[idx]

    print(
        f"{rank}. "
        f"Product ID: {product['product_id']} | "
        f"Category: {product['product_category']} | "
        f"Price: {product['unit_price']} | "
        f"Rating: {product['rating']} | "
        f"Score: {score:.4f}"
    )


print("\n" + "=" * 60)
print("FAISS PRODUCT KNOWLEDGE BASE COMPLETED")
print("=" * 60)