import os
import sys
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))

from rag_chatbot import faiss_search, ask_shopping_assistant


st.set_page_config(
    page_title="E-Commerce AI Assistant",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ E-Commerce AI Assistant")
st.caption("Recommendation System + Customer Segmentation + GenAI RAG")


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


@st.cache_data
def load_products():

    if not os.path.exists(PRODUCT_FILE):
        return pd.DataFrame()

    df = pd.read_csv(PRODUCT_FILE)

    if "product_id" in df.columns:
        df["product_id"] = df["product_id"].astype(str)

    if "product_category" in df.columns:
        df["product_category"] = (
            df["product_category"]
            .fillna("")
            .astype(str)
        )

    if "unit_price" in df.columns:
        df["unit_price"] = pd.to_numeric(
            df["unit_price"],
            errors="coerce"
        ).fillna(0)

    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(
            df["rating"],
            errors="coerce"
        ).fillna(0)

    return df


@st.cache_data
def load_segments():

    if not os.path.exists(SEGMENT_FILE):
        return pd.DataFrame()

    return pd.read_csv(SEGMENT_FILE)


products = load_products()
segments = load_segments()


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

    st.header("📊 E-Commerce Dashboard")

    if products.empty:

        st.error("products.csv not found.")

    else:

        total_products = len(products)

        if "customer_id" in segments.columns:
            total_customers = segments["customer_id"].nunique()
        else:
            total_customers = 0

        if "rating" in products.columns:
            avg_rating = products["rating"].mean()
        else:
            avg_rating = 0

        if "unit_price" in products.columns:
            avg_price = products["unit_price"].mean()
        else:
            avg_price = 0

        col1, col2, col3, col4 = st.columns(4)

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

        st.subheader("Top Product Categories")

        if "product_category" in products.columns:

            category_df = (
                products["product_category"]
                .value_counts()
                .head(10)
                .rename_axis("product_category")
                .reset_index(name="products")
            )

            st.bar_chart(
                category_df.set_index(
                    "product_category"
                )["products"]
            )

        st.info(
            "Cloud deployment uses the processed product "
            "catalog instead of the local MySQL database."
        )


# ==========================================================
# RECOMMENDATIONS
# ==========================================================

elif page == "🤖 Recommendations":

    st.header("🤖 Product Recommendations")

    st.write(
        "Select a product to find similar products."
    )

    if products.empty:

        st.error("products.csv not found.")

    else:

        display_df = products[
            [
                "product_id",
                "product_category",
                "unit_price",
                "rating"
            ]
        ].copy()

        display_df["display"] = (
            display_df["product_id"].astype(str)
            + " | Category "
            + display_df["product_category"].astype(str)
            + " | ₹"
            + display_df["unit_price"]
            .round(2)
            .astype(str)
        )

        selected = st.selectbox(
            "Select Product",
            display_df["display"].tolist()
        )

        selected_product_id = selected.split(
            " | "
        )[0]

        if st.button("Get Recommendations"):

            selected_row = display_df[
                display_df["product_id"].astype(str)
                == selected_product_id
            ]

            if selected_row.empty:

                st.warning(
                    "Selected product was not found."
                )

            else:

                row = selected_row.iloc[0]

                query = (
                    f"Product ID: {row['product_id']}. "
                    f"Category: {row['product_category']}. "
                    f"Price: {row['unit_price']}. "
                    f"Rating: {row['rating']}"
                )

                try:

                    results = faiss_search(
                        query,
                        top_k=10
                    )

                    if results:

                        rec_df = pd.DataFrame(
                            results
                        )

                        if "product_id" in rec_df.columns:

                            rec_df["product_id"] = (
                                rec_df["product_id"]
                                .astype(str)
                            )

                            rec_df = rec_df[
                                rec_df["product_id"]
                                != selected_product_id
                            ]

                        rec_df = rec_df.head(5)

                        st.subheader(
                            "Content-Based Recommendations"
                        )

                        if not rec_df.empty:

                            st.dataframe(
                                rec_df,
                                width="stretch"
                            )

                        else:

                            st.info(
                                "No similar products found."
                            )

                    else:

                        st.info(
                            "No recommendations found."
                        )

                    st.subheader(
                        "Collaborative Recommendations"
                    )

                    st.info(
                        "Collaborative model artifacts are "
                        "not deployed. FAISS content-based "
                        "recommendations are available in "
                        "the cloud."
                    )

                    st.subheader(
                        "Hybrid Recommendations"
                    )

                    st.info(
                        "Hybrid recommendations require the "
                        "collaborative model artifact."
                    )

                except Exception as e:

                    st.error(
                        f"Recommendation error: {e}"
                    )


# ==========================================================
# CUSTOMER SEGMENTS
# ==========================================================

elif page == "👥 Customer Segments":

    st.header("👥 Customer Segmentation")

    if segments.empty:

        st.warning(
            "customer_segments.csv not found."
        )

    elif "segment" not in segments.columns:

        st.error(
            "The customer_segments.csv file does not "
            "contain a segment column."
        )

    else:

        st.subheader(
            "Customer Segment Distribution"
        )

        st.bar_chart(
            segments["segment"]
            .value_counts()
            .sort_index()
        )

        st.subheader(
            "Segment Summary"
        )

        if {
            "customer_id",
            "frequency",
            "monetary",
            "recency"
        }.issubset(segments.columns):

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
                .to_frame("customers")
            )

        st.dataframe(
            summary,
            width="stretch"
        )

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

                    result = ask_shopping_assistant(
                        question.strip()
                    )

                    if isinstance(result, dict):

                        st.subheader(
                            "🤖 AI Answer"
                        )

                        st.write(
                            result.get(
                                "answer",
                                "No answer returned."
                            )
                        )

                        if result.get("sources"):

                            st.subheader(
                                "📚 Retrieved Products"
                            )

                            st.dataframe(
                                pd.DataFrame(
                                    result["sources"]
                                ),
                                width="stretch"
                            )

                    elif (
                        isinstance(result, str)
                        and result.strip()
                    ):

                        st.subheader(
                            "🤖 AI Answer"
                        )

                        st.write(result)

                    else:

                        st.info(
                            "The assistant completed the "
                            "request but did not return "
                            "displayable text."
                        )

                except Exception as e:

                    st.error(
                        f"AI Assistant error: {e}"
                    )