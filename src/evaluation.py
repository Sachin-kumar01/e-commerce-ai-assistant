import pandas as pd
import numpy as np
import joblib


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/processed"
MODEL_PATH = "models"


# ============================================================
# LOAD PURCHASE DATA
# ============================================================

def load_purchase_data():

    print("Loading orders...")

    orders = pd.read_csv(
        f"{DATA_PATH}/orders.csv"
    )

    print("Loading order items...")

    order_items = pd.read_csv(
        f"{DATA_PATH}/order_items.csv"
    )

    # Only purchased orders
    orders = orders[
        orders["purchased"] == 1
    ].copy()

    interactions = orders[
        [
            "order_id",
            "customer_id",
            "visit_date"
        ]
    ].merge(
        order_items[
            [
                "order_id",
                "product_id"
            ]
        ],
        on="order_id",
        how="inner"
    )

    interactions["visit_date"] = pd.to_datetime(
        interactions["visit_date"]
    )

    return interactions


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def create_train_test(interactions):

    interactions = interactions.sort_values(
        [
            "customer_id",
            "visit_date"
        ]
    )

    train_rows = []
    test_rows = []

    for customer_id, group in interactions.groupby(
        "customer_id"
    ):

        products = group[
            "product_id"
        ].unique()

        # Customer must have at least
        # two different purchased products
        if len(products) < 2:
            continue

        # Last purchased product = test item
        test_product = products[-1]

        # Earlier purchased products = train
        train_products = products[:-1]

        for product in train_products:

            train_rows.append(
                [
                    customer_id,
                    product
                ]
            )

        test_rows.append(
            [
                customer_id,
                test_product
            ]
        )

    train = pd.DataFrame(
        train_rows,
        columns=[
            "customer_id",
            "product_id"
        ]
    )

    test = pd.DataFrame(
        test_rows,
        columns=[
            "customer_id",
            "product_id"
        ]
    )

    return train, test


# ============================================================
# PRECISION @ K
# ============================================================

def precision_at_k(
    recommendations,
    actual,
    k=5
):

    recommendations = recommendations[:k]

    if len(recommendations) == 0:
        return 0.0

    hits = len(
        set(recommendations)
        &
        set(actual)
    )

    return hits / k


# ============================================================
# RECALL @ K
# ============================================================

def recall_at_k(
    recommendations,
    actual,
    k=5
):

    recommendations = recommendations[:k]

    if len(actual) == 0:
        return 0.0

    hits = len(
        set(recommendations)
        &
        set(actual)
    )

    return hits / len(actual)


# ============================================================
# HIT RATE @ K
# ============================================================

def hit_rate_at_k(
    recommendations,
    actual,
    k=5
):

    recommendations = recommendations[:k]

    hits = len(
        set(recommendations)
        &
        set(actual)
    )

    return int(hits > 0)


# ============================================================
# HELPER:
# CONVERT SPARSE / DENSE ROW TO NUMPY ARRAY
# ============================================================

def get_similarity_scores(
    similarity,
    index
):

    # Sparse matrix
    if hasattr(similarity, "getrow"):

        scores = similarity.getrow(index)

        scores = scores.toarray().flatten()

    else:

        scores = np.asarray(
            similarity[index]
        ).flatten()

    return scores


# ============================================================
# LOAD CONTENT MODEL
# ============================================================

def load_content_model():

    print("Loading Content-Based model...")

    similarity = joblib.load(
        f"{MODEL_PATH}/content_similarity.pkl"
    )

    products = joblib.load(
        f"{MODEL_PATH}/content_products.pkl"
    )

    return similarity, products


# ============================================================
# CONTENT-BASED RECOMMENDATIONS
# ============================================================

def content_recommendations(
    product_id,
    similarity,
    products,
    k=5
):

    product_ids = products[
        "product_id"
    ].values

    # Product doesn't exist
    if product_id not in product_ids:

        return []

    # Product index
    index = np.where(
        product_ids == product_id
    )[0][0]

    # Get similarity scores
    scores = get_similarity_scores(
        similarity,
        index
    )

    # Rank products by similarity
    ranked_indices = np.argsort(
        scores
    )[::-1]

    recommendations = []

    for idx in ranked_indices:

        recommended_product = int(
            product_ids[idx]
        )

        # Don't recommend same product
        if recommended_product == product_id:
            continue

        score = float(
            scores[idx]
        )

        # Ignore zero similarity
        if score <= 0:
            continue

        recommendations.append(
            recommended_product
        )

        if len(recommendations) >= k:
            break

    return recommendations


# ============================================================
# LOAD COLLABORATIVE MODEL
# ============================================================

def load_collaborative_model():

    print(
        "Loading Collaborative Filtering model..."
    )

    similarity = joblib.load(
        f"{MODEL_PATH}/collaborative_similarity.pkl"
    )

    product_ids = joblib.load(
        f"{MODEL_PATH}/collaborative_product_ids.pkl"
    )

    return similarity, product_ids


# ============================================================
# COLLABORATIVE RECOMMENDATIONS
# ============================================================

def collaborative_recommendations(
    product_id,
    similarity,
    product_ids,
    k=5
):

    product_ids = np.asarray(
        product_ids
    )

    # Product doesn't exist
    if product_id not in product_ids:

        return []

    # Product index
    index = np.where(
        product_ids == product_id
    )[0][0]

    # Get similarity scores
    scores = get_similarity_scores(
        similarity,
        index
    )

    # Rank products
    ranked_indices = np.argsort(
        scores
    )[::-1]

    recommendations = []

    for idx in ranked_indices:

        recommended_product = int(
            product_ids[idx]
        )

        # Don't recommend same product
        if recommended_product == product_id:
            continue

        score = float(
            scores[idx]
        )

        # Ignore zero similarity
        if score <= 0:
            continue

        recommendations.append(
            recommended_product
        )

        if len(recommendations) >= k:
            break

    return recommendations


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    name,
    train,
    test,
    recommendation_function,
    model_data
):

    precision_scores = []
    recall_scores = []
    hit_scores = []

    evaluated = 0

    print(
        f"\nEvaluating {name} model..."
    )

    for _, row in test.iterrows():

        customer_id = row[
            "customer_id"
        ]

        actual_product = int(
            row["product_id"]
        )

        # Customer's training history
        customer_history = train[
            train["customer_id"]
            ==
            customer_id
        ][
            "product_id"
        ].tolist()

        if len(customer_history) == 0:
            continue

        recommendations = []

        # Generate recommendations
        # from products already purchased
        for product in customer_history:

            recs = recommendation_function(
                product,
                *model_data,
                k=5
            )

            for rec in recs:

                # Don't duplicate recommendations
                if rec not in recommendations:

                    recommendations.append(
                        rec
                    )

                if len(recommendations) >= 5:
                    break

            if len(recommendations) >= 5:
                break

        if len(recommendations) == 0:
            continue

        actual = [
            actual_product
        ]

        precision_scores.append(
            precision_at_k(
                recommendations,
                actual,
                k=5
            )
        )

        recall_scores.append(
            recall_at_k(
                recommendations,
                actual,
                k=5
            )
        )

        hit_scores.append(
            hit_rate_at_k(
                recommendations,
                actual,
                k=5
            )
        )

        evaluated += 1

    # No evaluations
    if evaluated == 0:

        return {
            "model": name,
            "precision": 0.0,
            "recall": 0.0,
            "hit_rate": 0.0,
            "evaluated": 0
        }

    return {
        "model": name,

        "precision": float(
            np.mean(
                precision_scores
            )
        ),

        "recall": float(
            np.mean(
                recall_scores
            )
        ),

        "hit_rate": float(
            np.mean(
                hit_scores
            )
        ),

        "evaluated": evaluated
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("RECOMMENDATION SYSTEM EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print(
        "\nLoading purchase interactions..."
    )

    interactions = load_purchase_data()

    print(
        f"Purchase interactions: "
        f"{len(interactions)}"
    )

    print(
        f"Unique customers: "
        f"{interactions['customer_id'].nunique()}"
    )

    print(
        f"Unique products: "
        f"{interactions['product_id'].nunique()}"
    )

    # --------------------------------------------------------
    # TRAIN / TEST
    # --------------------------------------------------------

    print(
        "\nCreating train/test split..."
    )

    train, test = create_train_test(
        interactions
    )

    print(
        f"Training interactions: "
        f"{len(train)}"
    )

    print(
        f"Test customers: "
        f"{len(test)}"
    )

    # --------------------------------------------------------
    # CONTENT MODEL
    # --------------------------------------------------------

    try:

        content_similarity, content_products = (
            load_content_model()
        )

        print(
            "Content model loaded successfully."
        )

        content_result = evaluate_model(
            name="Content-Based",
            train=train,
            test=test,
            recommendation_function=(
                content_recommendations
            ),
            model_data=(
                content_similarity,
                content_products
            )
        )

    except FileNotFoundError:

        print(
            "\nContent model files not found."
        )

        content_result = {
            "model": "Content-Based",
            "precision": 0.0,
            "recall": 0.0,
            "hit_rate": 0.0,
            "evaluated": 0
        }

    # --------------------------------------------------------
    # COLLABORATIVE MODEL
    # --------------------------------------------------------

    try:

        collaborative_similarity, collaborative_product_ids = (
            load_collaborative_model()
        )

        print(
            "Collaborative model loaded successfully."
        )

        collaborative_result = evaluate_model(
            name="Collaborative",
            train=train,
            test=test,
            recommendation_function=(
                collaborative_recommendations
            ),
            model_data=(
                collaborative_similarity,
                collaborative_product_ids
            )
        )

    except FileNotFoundError:

        print(
            "\nCollaborative model files not found."
        )

        collaborative_result = {
            "model": "Collaborative",
            "precision": 0.0,
            "recall": 0.0,
            "hit_rate": 0.0,
            "evaluated": 0
        }

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = pd.DataFrame(
        [
            content_result,
            collaborative_result
        ]
    )

    print("\n")
    print("=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)

    print(
        results[
            [
                "model",
                "precision",
                "recall",
                "hit_rate",
                "evaluated"
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    output_path = (
        f"{MODEL_PATH}/evaluation_results.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\n")
    print(
        "Evaluation results saved to:"
    )

    print(output_path)

    print("\n")
    print(
        "=" * 70
    )

    print(
        "EVALUATION COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )