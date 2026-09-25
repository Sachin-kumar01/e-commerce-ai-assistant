import os
import pickle
import numpy as np
import faiss

from rag_chatbot import (
    create_product_documents,
    create_embeddings
)


INDEX_PATH = "embeddings/products.index"
METADATA_PATH = "embeddings/products_metadata.pkl"


def build_index():

    print("\n================================")
    print("BUILDING PRODUCT FAISS INDEX")
    print("================================")

    documents = create_product_documents()

    print(
        f"\nTotal products: {len(documents)}"
    )

    texts = [
        document["text"]
        for document in documents
    ]

    print("\nCreating embeddings...")

    embeddings = create_embeddings(
        texts,
        batch_size=50
    )

    embedding_matrix = np.array(
        embeddings,
        dtype="float32"
    )

    print(
        "\nEmbedding shape:",
        embedding_matrix.shape
    )

    # Normalize vectors
    faiss.normalize_L2(
        embedding_matrix
    )

    dimension = (
        embedding_matrix.shape[1]
    )

    # Cosine similarity using inner product
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embedding_matrix
    )

    os.makedirs(
        "embeddings",
        exist_ok=True
    )

    faiss.write_index(
        index,
        INDEX_PATH
    )

    with open(
        METADATA_PATH,
        "wb"
    ) as file:

        pickle.dump(
            documents,
            file
        )

    print("\n================================")
    print("FAISS INDEX CREATED")
    print("================================")

    print(
        "Vectors:",
        index.ntotal
    )

    print(
        "Dimension:",
        dimension
    )

    print(
        "Index:",
        INDEX_PATH
    )

    print(
        "Metadata:",
        METADATA_PATH
    )


if __name__ == "__main__":
    build_index()