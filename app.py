import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib


# ==========================================================
# BASE DIRECTORY
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_DIR = os.path.join(
    BASE_DIR,
    "src"
)

sys.path.append(SRC_DIR)


# ==========================================================
# IMPORT AI / FAISS FUNCTIONS
# ==========================================================

from rag_chatbot import (
    faiss_search,
    ask_shopping_assistant
)


# ==========================================================
# STREAMLIT CONFIG
# ==========================================================

st.set_page_config(
    page_title="E-Commerce AI Assistant",
    page_icon="🛍️",
    layout="wide"
)


# ==========================================================
# TITLE
# ==========================================================

st.title(
    "🛍️ E-Commerce AI Assistant"
)

st.caption(
    "Recommendation System + Customer Segmentation + GenAI RAG"
)


# ==========================================================
# FILE PATHS
# ==========================================================

PRODUCT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "products.csv"
)

SEGMENT_FILE = os.path.join(
    BASE_DIR,
    "models",
    "customer_segments.csv"
)

COLLAB_SIMILARITY_FILE = os.path.join(
    BASE_DIR,
    "models",
    "collaborative_similarity.pkl"
)

COLLAB_PRODUCT_IDS_FILE = os.path.join(
    BASE_DIR,
    "models",
    "collaborative_product_ids.pkl"
)


# ==========================================================
# LOAD PRODUCTS
# ==========================================================

@st.cache_data
def load_products():

    if not os.path.exists(PRODUCT_FILE):

        return pd.DataFrame()

    df = pd.read_csv(
        PRODUCT_FILE
    )

    # Product ID
    if "product_id" in df.columns:

        df["product_id"] = (
            df["product_id"]
            .astype(str)
        )

    # Category
    if "product_category" in df.columns:

        df["product_category"] = (
            df["product_category"]
            .fillna("")
            .astype(str)
        )

    # Price
    if "unit_price" in df.columns:

        df["unit_price"] = pd.to_numeric(
            df["unit_price"],
            errors="coerce"
        ).fillna(0)

    # Rating
    if "rating" in df.columns:

        df["rating"] = pd.to_numeric(
            df["rating"],
            errors="coerce"
        ).fillna(0)

    return df


# ==========================================================
# LOAD CUSTOMER SEGMENTS
# ==========================================================

@st.cache_data
def load_segments():

    if not os.path.exists(
        SEGMENT_FILE
    ):

        return pd.DataFrame()

    return pd.read_csv(
        SEGMENT_FILE
    )


# ==========================================================
# LOAD COLLABORATIVE MODEL
# ==========================================================

@st.cache_resource
def load_collaborative_model():

    if not os.path.exists(
        COLLAB_SIMILARITY_FILE
    ):

        return None, None

    if not os.path.exists(
        COLLAB_PRODUCT_IDS_FILE
    ):

        return None, None

    try:

        # IMPORTANT:
        # These files were created using joblib.dump()
        # Therefore they must be loaded using joblib.load()

        collaborative_similarity = joblib.load(
            COLLAB_SIMILARITY_FILE
        )

        collaborative_product_ids = joblib.load(
            COLLAB_PRODUCT_IDS_FILE
        )

        # Convert sparse matrix if necessary

        if hasattr(
            collaborative_similarity,
            "toarray"
        ):

            collaborative_similarity = (
                collaborative_similarity.toarray()
            )

        collaborative_similarity = np.asarray(
            collaborative_similarity
        )

        collaborative_product_ids = [
            str(product_id)
            for product_id in collaborative_product_ids
        ]

        return (
            collaborative_similarity,
            collaborative_product_ids
        )

    except Exception as e:

        st.error(
            f"Collaborative model loading error: {e}"
        )

        return None, None


# ==========================================================
# COLLABORATIVE RECOMMENDATION
# ==========================================================

def get_collaborative_recommendations(
    product_id,
    similarity_matrix,
    product_ids,
    top_k=5
):

    if (
        similarity_matrix is None
        or product_ids is None
    ):

        return pd.DataFrame(
            columns=[
                "product_id",
                "collaborative_score"
            ]
        )

    product_id = str(
        product_id
    )

    # Check selected product

    if product_id not in product_ids:

        return pd.DataFrame(
            columns=[
                "product_id",
                "collaborative_score"
            ]
        )

    # Product index

    product_index = product_ids.index(
        product_id
    )

    # Similarity scores

    scores = np.asarray(
        similarity_matrix[
            product_index
        ]
    ).flatten()

    # Sort highest score first

    ranked_indices = np.argsort(
        scores
    )[::-1]

    recommendations = []

    for index in ranked_indices:

        # Don't recommend selected product

        if index == product_index:

            continue

        score = float(
            scores[index]
        )

        # Ignore zero similarity

        if score <= 0:

            continue

        recommendations.append(
            {
                "product_id": product_ids[index],
                "collaborative_score": score
            }
        )

        if len(
            recommendations
        ) >= top_k:

            break

    return pd.DataFrame(
        recommendations
    )


# ==========================================================
# NORMALIZE SCORE
# ==========================================================

def normalize_score(
    series
):

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)

    if len(series) == 0:

        return series

    min_value = series.min()

    max_value = series.max()

    if max_value == min_value:

        if max_value == 0:

            return pd.Series(
                0.0,
                index=series.index
            )

        return pd.Series(
            1.0,
            index=series.index
        )

    return (
        (series - min_value)
        /
        (max_value - min_value)
    )


# ==========================================================
# HYBRID RECOMMENDATION
# ==========================================================

def create_hybrid_recommendations(
    content_results,
    collaborative_results,
    top_k=5
):

    # ------------------------------------------------------
    # CONTENT RESULTS
    # ------------------------------------------------------

    if content_results is None:

        content_results = pd.DataFrame()

    else:

        content_results = pd.DataFrame(
            content_results
        ).copy()

    if not content_results.empty:

        if "product_id" in content_results.columns:

            content_results[
                "product_id"
            ] = (
                content_results[
                    "product_id"
                ].astype(str)
            )

        # FAISS may return similarity
        # or similarity_score

        if "similarity_score" in (
            content_results.columns
        ):

            content_results[
                "content_score"
            ] = pd.to_numeric(
                content_results[
                    "similarity_score"
                ],
                errors="coerce"
            ).fillna(0)

        elif "similarity" in (
            content_results.columns
        ):

            content_results[
                "content_score"
            ] = pd.to_numeric(
                content_results[
                    "similarity"
                ],
                errors="coerce"
            ).fillna(0)

        elif "score" in (
            content_results.columns
        ):

            content_results[
                "content_score"
            ] = pd.to_numeric(
                content_results[
                    "score"
                ],
                errors="coerce"
            ).fillna(0)

        else:

            content_results[
                "content_score"
            ] = 0.0

        content_results = (
            content_results[
                [
                    "product_id",
                    "content_score"
                ]
            ]
        )

    else:

        content_results = pd.DataFrame(
            columns=[
                "product_id",
                "content_score"
            ]
        )


    # ------------------------------------------------------
    # COLLABORATIVE RESULTS
    # ------------------------------------------------------

    if collaborative_results is None:

        collaborative_results = pd.DataFrame()

    else:

        collaborative_results = (
            pd.DataFrame(
                collaborative_results
            ).copy()
        )

    if not collaborative_results.empty:

        collaborative_results[
            "product_id"
        ] = (
            collaborative_results[
                "product_id"
            ].astype(str)
        )

        collaborative_results[
            "collaborative_score"
        ] = pd.to_numeric(
            collaborative_results[
                "collaborative_score"
            ],
            errors="coerce"
        ).fillna(0)

        collaborative_results = (
            collaborative_results[
                [
                    "product_id",
                    "collaborative_score"
                ]
            ]
        )

    else:

        collaborative_results = pd.DataFrame(
            columns=[
                "product_id",
                "collaborative_score"
            ]
        )


    # ------------------------------------------------------
    # MERGE
    # ------------------------------------------------------

    hybrid = pd.merge(
        content_results,
        collaborative_results,
        on="product_id",
        how="outer"
    )

    if hybrid.empty:

        return hybrid


    # ------------------------------------------------------
    # FILL MISSING SCORES
    # ------------------------------------------------------

    hybrid[
        "content_score"
    ] = hybrid[
        "content_score"
    ].fillna(0)

    hybrid[
        "collaborative_score"
    ] = hybrid[
        "collaborative_score"
    ].fillna(0)


    # ------------------------------------------------------
    # NORMALIZE
    # ------------------------------------------------------

    hybrid[
        "content_normalized"
    ] = normalize_score(
        hybrid[
            "content_score"
        ]
    )

    hybrid[
        "collaborative_normalized"
    ] = normalize_score(
        hybrid[
            "collaborative_score"
        ]
    )


    # ------------------------------------------------------
    # HYBRID SCORE
    #
    # 60% Content-Based
    # 40% Collaborative
    # ------------------------------------------------------

    hybrid[
        "hybrid_score"
    ] = (
        0.6
        * hybrid[
            "content_normalized"
        ]
        +
        0.4
        * hybrid[
            "collaborative_normalized"
        ]
    )


    # ------------------------------------------------------
    # SORT
    # ------------------------------------------------------

    hybrid = (
        hybrid
        .sort_values(
            "hybrid_score",
            ascending=False
        )
        .head(top_k)
    )


    return hybrid


# ==========================================================
# LOAD DATA
# ==========================================================

products = load_products()

segments = load_segments()

(
    collaborative_similarity,
    collaborative_product_ids
) = load_collaborative_model()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title(
    "Navigation"
)

page = st.sidebar.radio(
    "Go to",
    [
        "📊 Dashboard",
        "🤖 Recommendations",
        "👥 Customer Segments",
        "💬 AI Shopping Assistant"
    ]
)


# ==========================================================
# DASHBOARD
# ==========================================================

if page == "📊 Dashboard":

    st.header(
        "📊 E-Commerce Dashboard"
    )

    if products.empty:

        st.error(
            "products.csv not found."
        )

    else:

        # ----------------------------------------------
        # CUSTOMERS
        # ----------------------------------------------

        if (
            not segments.empty
            and "customer_id" in segments.columns
        ):

            total_customers = (
                segments[
                    "customer_id"
                ].nunique()
            )

        else:

            total_customers = 0


        # ----------------------------------------------
        # PRODUCTS
        # ----------------------------------------------

        total_products = len(
            products
        )


        # ----------------------------------------------
        # AVG RATING
        # ----------------------------------------------

        if "rating" in products.columns:

            avg_rating = (
                products[
                    "rating"
                ].mean()
            )

        else:

            avg_rating = 0


        # ----------------------------------------------
        # AVG PRICE
        # ----------------------------------------------

        if "unit_price" in products.columns:

            avg_price = (
                products[
                    "unit_price"
                ].mean()
            )

        else:

            avg_price = 0


        # ----------------------------------------------
        # METRICS
        # ----------------------------------------------

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "Customers",
            f"{total_customers:,}"
        )

        col2.metric(
            "Products",
            f"{total_products:,}"
        )

        col3.metric(
            "Avg Rating",
            f"{avg_rating:.2f}"
        )

        col4.metric(
            "Avg Price",
            f"₹ {avg_price:,.2f}"
        )


        st.divider()


        # ----------------------------------------------
        # CATEGORY CHART
        # ----------------------------------------------

        st.subheader(
            "Top Product Categories"
        )

        if "product_category" in (
            products.columns
        ):

            category_df = (
                products[
                    "product_category"
                ]
                .value_counts()
                .head(10)
                .rename_axis(
                    "product_category"
                )
                .reset_index(
                    name="products"
                )
            )

            category_df[
                "product_category"
            ] = (
                "Category "
                +
                category_df[
                    "product_category"
                ].astype(str)
            )

            st.bar_chart(
                category_df.set_index(
                    "product_category"
                )[
                    "products"
                ]
            )

        else:

            st.info(
                "No category data available."
            )


        st.info(
            "Cloud deployment uses the processed "
            "product catalog instead of the local "
            "MySQL database."
        )


# ==========================================================
# RECOMMENDATIONS
# ==========================================================

elif page == "🤖 Recommendations":

    st.header(
        "🤖 Product Recommendations"
    )

    st.write(
        "Select a product to find similar products "
        "using Content-Based, Collaborative and "
        "Hybrid recommendation models."
    )


    if products.empty:

        st.error(
            "products.csv not found."
        )

    else:

        # ----------------------------------------------
        # PRODUCT DISPLAY
        # ----------------------------------------------

        product_df = products[
            [
                "product_id",
                "product_category",
                "unit_price",
                "rating"
            ]
        ].copy()


        product_df[
            "display"
        ] = (
            product_df[
                "product_id"
            ].astype(str)
            +
            " | Category "
            +
            product_df[
                "product_category"
            ].astype(str)
            +
            " | ₹"
            +
            product_df[
                "unit_price"
            ].round(2).astype(str)
        )


        # ----------------------------------------------
        # SELECT PRODUCT
        # ----------------------------------------------

        selected = st.selectbox(
            "Select Product",
            product_df[
                "display"
            ].tolist()
        )


        selected_product_id = (
            selected.split(
                " | "
            )[0]
        )


        # ----------------------------------------------
        # BUTTON
        # ----------------------------------------------

        if st.button(
            "Get Recommendations"
        ):

            selected_row = product_df[
                product_df[
                    "product_id"
                ].astype(str)
                ==
                selected_product_id
            ]


            if selected_row.empty:

                st.warning(
                    "Selected product was not found."
                )

            else:

                row = selected_row.iloc[0]


                # ==================================================
                # CONTENT-BASED USING FAISS
                # ==================================================

                try:

                    query = (
                        f"Product ID: "
                        f"{row['product_id']}. "
                        f"Category: "
                        f"{row['product_category']}. "
                        f"Price: "
                        f"{row['unit_price']}. "
                        f"Rating: "
                        f"{row['rating']}"
                    )


                    content_results = faiss_search(
                        query,
                        top_k=10
                    )


                    content_results = pd.DataFrame(
                        content_results
                    )


                    if not content_results.empty:

                        if "product_id" in (
                            content_results.columns
                        ):

                            content_results[
                                "product_id"
                            ] = (
                                content_results[
                                    "product_id"
                                ].astype(str)
                            )


                            # Remove selected product

                            content_results = (
                                content_results[
                                    content_results[
                                        "product_id"
                                    ]
                                    !=
                                    selected_product_id
                                ]
                            )


                        content_results = (
                            content_results.head(5)
                        )


                    st.subheader(
                        "🔎 Content-Based Recommendations"
                    )


                    if not content_results.empty:

                        st.dataframe(
                            content_results,
                            width="stretch"
                        )

                    else:

                        st.info(
                            "No content-based "
                            "recommendations found."
                        )


                except Exception as e:

                    st.error(
                        f"Content recommendation error: {e}"
                    )

                    content_results = (
                        pd.DataFrame()
                    )


                # ==================================================
                # COLLABORATIVE
                # ==================================================

                st.subheader(
                    "👥 Collaborative Recommendations"
                )


                collaborative_results = (
                    get_collaborative_recommendations(
                        selected_product_id,
                        collaborative_similarity,
                        collaborative_product_ids,
                        top_k=10
                    )
                )


                if not collaborative_results.empty:

                    collaborative_display = (
                        collaborative_results.merge(
                            products[
                                [
                                    "product_id",
                                    "product_category",
                                    "unit_price",
                                    "rating"
                                ]
                            ],
                            on="product_id",
                            how="left"
                        )
                    )


                    collaborative_display = (
                        collaborative_display[
                            [
                                "product_id",
                                "product_category",
                                "unit_price",
                                "rating",
                                "collaborative_score"
                            ]
                        ]
                        .head(5)
                    )


                    collaborative_display[
                        "collaborative_score"
                    ] = (
                        collaborative_display[
                            "collaborative_score"
                        ].round(4)
                    )


                    st.dataframe(
                        collaborative_display,
                        width="stretch"
                    )

                else:

                    if (
                        collaborative_similarity
                        is None
                    ):

                        st.warning(
                            "Collaborative model is "
                            "not available. Make sure "
                            "collaborative model files "
                            "are present in the models folder."
                        )

                    else:

                        st.info(
                            "No collaborative "
                            "recommendations available "
                            "for this product."
                        )


                # ==================================================
                # HYBRID
                # ==================================================

                st.subheader(
                    "🔀 Hybrid Recommendations"
                )


                hybrid_results = (
                    create_hybrid_recommendations(
                        content_results,
                        collaborative_results,
                        top_k=5
                    )
                )


                if not hybrid_results.empty:

                    hybrid_display = (
                        hybrid_results.merge(
                            products[
                                [
                                    "product_id",
                                    "product_category",
                                    "unit_price",
                                    "rating"
                                ]
                            ],
                            on="product_id",
                            how="left"
                        )
                    )


                    hybrid_display = (
                        hybrid_display[
                            [
                                "product_id",
                                "product_category",
                                "unit_price",
                                "rating",
                                "content_score",
                                "collaborative_score",
                                "hybrid_score"
                            ]
                        ]
                    )


                    hybrid_display[
                        "content_score"
                    ] = (
                        hybrid_display[
                            "content_score"
                        ].round(4)
                    )


                    hybrid_display[
                        "collaborative_score"
                    ] = (
                        hybrid_display[
                            "collaborative_score"
                        ].round(4)
                    )


                    hybrid_display[
                        "hybrid_score"
                    ] = (
                        hybrid_display[
                            "hybrid_score"
                        ].round(4)
                    )


                    st.dataframe(
                        hybrid_display,
                        width="stretch"
                    )

                else:

                    st.info(
                        "No hybrid recommendations "
                        "could be generated."
                    )


# ==========================================================
# CUSTOMER SEGMENTS
# ==========================================================

elif page == "👥 Customer Segments":

    st.header(
        "👥 Customer Segmentation"
    )


    if segments.empty:

        st.warning(
            "customer_segments.csv not found."
        )

        st.info(
            "Run customer segmentation first."
        )

    elif "segment" not in (
        segments.columns
    ):

        st.error(
            "The customer_segments.csv file "
            "does not contain a segment column."
        )

    else:

        # ----------------------------------------------
        # DISTRIBUTION
        # ----------------------------------------------

        st.subheader(
            "Customer Segment Distribution"
        )


        segment_counts = (
            segments[
                "segment"
            ]
            .value_counts()
            .sort_index()
        )


        st.bar_chart(
            segment_counts
        )


        # ----------------------------------------------
        # SUMMARY
        # ----------------------------------------------

        st.subheader(
            "Segment Summary"
        )


        if {
            "customer_id",
            "frequency",
            "monetary",
            "recency"
        }.issubset(
            segments.columns
        ):

            summary = (
                segments
                .groupby("segment")
                .agg(
                    customers=(
                        "customer_id",
                        "count"
                    ),
                    avg_recency=(
                        "recency",
                        "mean"
                    ),
                    avg_frequency=(
                        "frequency",
                        "mean"
                    ),
                    avg_monetary=(
                        "monetary",
                        "mean"
                    )
                )
                .round(2)
            )

        else:

            summary = (
                segments
                .groupby("segment")
                .size()
                .to_frame(
                    "customers"
                )
            )


        st.dataframe(
            summary,
            width="stretch"
        )


        # ----------------------------------------------
        # CUSTOMER DATA
        # ----------------------------------------------

        st.subheader(
            "Customer Segment Data"
        )


        st.dataframe(
            segments,
            width="stretch"
        )


# ==========================================================
# AI SHOPPING ASSISTANT
# ==========================================================

elif page == "💬 AI Shopping Assistant":

    st.header(
        "💬 AI Shopping Assistant"
    )

    st.write(
        "Ask questions about products in the catalog."
    )


    # ----------------------------------------------
    # FORM
    # ----------------------------------------------

    with st.form(
        "shopping_assistant_form"
    ):

        question = st.text_input(
            "Ask your question",
            placeholder=(
                "Example: Show me products under 1000"
            )
        )


        submitted = st.form_submit_button(
            "Ask AI"
        )


    # ----------------------------------------------
    # PROCESS
    # ----------------------------------------------

    if submitted:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching products and generating answer..."
            ):

                try:

                    result = (
                        ask_shopping_assistant(
                            question.strip()
                        )
                    )


                    # ----------------------------------
                    # DICTIONARY RESPONSE
                    # ----------------------------------

                    if isinstance(
                        result,
                        dict
                    ):

                        st.subheader(
                            "🤖 AI Answer"
                        )


                        st.write(
                            result.get(
                                "answer",
                                "No answer returned."
                            )
                        )


                        if result.get(
                            "sources"
                        ):

                            st.subheader(
                                "📚 Retrieved Products"
                            )


                            st.dataframe(
                                pd.DataFrame(
                                    result[
                                        "sources"
                                    ]
                                ),
                                width="stretch"
                            )


                    # ----------------------------------
                    # STRING RESPONSE
                    # ----------------------------------

                    elif isinstance(
                        result,
                        str
                    ) and result.strip():

                        st.subheader(
                            "🤖 AI Answer"
                        )


                        st.write(
                            result
                        )


                    else:

                        st.info(
                            "The assistant completed "
                            "the request but did not return "
                            "displayable text."
                        )


                except Exception as e:

                    st.error(
                        f"AI Assistant error: {e}"
                    )