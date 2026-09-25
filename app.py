import os
import sys
import io
from contextlib import redirect_stdout
import pandas as pd
import streamlit as st
from sqlalchemy import text
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from database import engine
from rag_chatbot import ask_shopping_assistant
st.set_page_config(page_title="E-Commerce AI Assistant", page_icon="🛍️", layout="wide")
st.title("🛍️ E-Commerce AI Assistant")
st.caption("Recommendation System + Customer Segmentation + GenAI RAG")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🤖 Recommendations", "👥 Customer Segments", "💬 AI Shopping Assistant"])
if page == "📊 Dashboard":
    st.header("📊 E-Commerce Dashboard")
    try:
        customers = pd.read_sql(text("SELECT COUNT(*) AS total FROM customers"), engine)
        orders = pd.read_sql(text("SELECT COUNT(*) AS total FROM orders"), engine)
        products = pd.read_sql(text("SELECT COUNT(*) AS total FROM products"), engine)
        revenue = pd.read_sql(text("SELECT ROUND(SUM(revenue), 2) AS revenue FROM order_items"), engine)
        total_customers = int(customers.iloc[0]["total"])
        total_orders = int(orders.iloc[0]["total"])
        total_products = int(products.iloc[0]["total"])
        total_revenue = float(revenue.iloc[0]["revenue"] or 0)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Customers", f"{total_customers:,}")
        col2.metric("Orders", f"{total_orders:,}")
        col3.metric("Products", f"{total_products:,}")
        col4.metric("Revenue", f"₹ {total_revenue:,.2f}")
        st.divider()
        st.subheader("Top Product Categories")
        category_query = "SELECT product_category, COUNT(*) AS sales FROM products p JOIN order_items oi ON p.product_id = oi.product_id WHERE product_category IS NOT NULL GROUP BY product_category ORDER BY sales DESC LIMIT 10"
        category_df = pd.read_sql(text(category_query), engine)
        if not category_df.empty:
            st.bar_chart(category_df.set_index("product_category")["sales"])
        else:
            st.info("No category data available.")
    except Exception as e:
        st.error(f"Dashboard error: {e}")
elif page == "🤖 Recommendations":
    st.header("🤖 Product Recommendations")
    st.write("Select a product to find similar products.")
    try:
        query = "SELECT product_id, product_category FROM products WHERE product_category IS NOT NULL LIMIT 1000"
        product_df = pd.read_sql(text(query), engine)
        product_df["display"] = product_df["product_id"].astype(str) + " | Category " + product_df["product_category"].astype(str)
        selected = st.selectbox("Select Product", product_df["display"].tolist())
        selected_product_id = selected.split(" | ")[0]
        if st.button("Get Recommendations"):
            import joblib
            import numpy as np
            MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
            @st.cache_resource
            def load_saved_recommendation_models():
                content_similarity = joblib.load(os.path.join(MODEL_DIR, "content_similarity.pkl"))
                content_products = joblib.load(os.path.join(MODEL_DIR, "content_products.pkl"))
                collaborative_similarity = joblib.load(os.path.join(MODEL_DIR, "collaborative_similarity.pkl"))
                collaborative_product_ids = joblib.load(os.path.join(MODEL_DIR, "collaborative_product_ids.pkl"))
                return content_similarity, content_products, collaborative_similarity, collaborative_product_ids
            try:
                st.info("Loading saved recommendation models...")
                content_similarity, content_products, collaborative_similarity, collaborative_product_ids = load_saved_recommendation_models()
                content_product_ids = content_products["product_id"].astype(str).tolist()
                if selected_product_id not in content_product_ids:
                    content_results = pd.DataFrame()
                else:
                    content_index = content_product_ids.index(selected_product_id)
                    if hasattr(content_similarity, "getrow"):
                        scores = content_similarity.getrow(content_index).toarray().ravel()
                    else:
                        scores = np.asarray(content_similarity[content_index]).ravel()
                    ranked_indices = np.argsort(scores)[::-1]
                    top_indices = [i for i in ranked_indices if i != content_index][:5]
                    available_cols = [c for c in ["product_id", "product_category", "unit_price", "rating"] if c in content_products.columns]
                    content_results = content_products.iloc[top_indices][available_cols].copy()
                    content_results["similarity_score"] = scores[top_indices]
                st.subheader("Content-Based Recommendations")
                if not content_results.empty:
                    st.dataframe(content_results, width="stretch")
                else:
                    st.info("No content-based recommendations found.")
                collaborative_product_ids = [str(product_id) for product_id in collaborative_product_ids]
                if selected_product_id in collaborative_product_ids:
                    collaborative_index = collaborative_product_ids.index(selected_product_id)
                    if hasattr(collaborative_similarity, "getrow"):
                        collab_scores = collaborative_similarity.getrow(collaborative_index).toarray().ravel()
                    else:
                        collab_scores = np.asarray(collaborative_similarity[collaborative_index]).ravel()
                    ranked_collab_indices = np.argsort(collab_scores)[::-1]
                    collab_indices = [i for i in ranked_collab_indices if i != collaborative_index and collab_scores[i] > 0][:5]
                    collaborative_results = pd.DataFrame({"product_id": [collaborative_product_ids[i] for i in collab_indices], "collaborative_score": [float(collab_scores[i]) for i in collab_indices]})
                else:
                    collaborative_results = pd.DataFrame(columns=["product_id", "collaborative_score"])
                st.subheader("Collaborative Recommendations")
                if not collaborative_results.empty:
                    st.dataframe(collaborative_results, width="stretch")
                else:
                    st.info("No collaborative recommendations available for this product.")
                if not content_results.empty:
                    hybrid_results = content_results[["product_id", "similarity_score"]].copy()
                    hybrid_results["product_id"] = hybrid_results["product_id"].astype(str)
                    hybrid_results = hybrid_results.rename(columns={"similarity_score": "content_score"})
                    if not collaborative_results.empty:
                        collaborative_results["product_id"] = collaborative_results["product_id"].astype(str)
                        hybrid_results = hybrid_results.merge(collaborative_results, on="product_id", how="left")
                    else:
                        hybrid_results["collaborative_score"] = 0.0
                    hybrid_results["collaborative_score"] = hybrid_results["collaborative_score"].fillna(0.0)
                    hybrid_results["hybrid_score"] = 0.6 * hybrid_results["content_score"] + 0.4 * hybrid_results["collaborative_score"]
                    hybrid_results = hybrid_results.sort_values("hybrid_score", ascending=False).head(5)
                    st.subheader("Hybrid Recommendations")
                    st.dataframe(hybrid_results, width="stretch")
            except Exception as e:
                st.error(f"Recommendation model error: {e}")
    except Exception as e:
        st.error(f"Recommendation page error: {e}")
elif page == "👥 Customer Segments":
    st.header("👥 Customer Segmentation")
    segment_file = os.path.join(os.path.dirname(__file__), "models", "customer_segments.csv")
    if not os.path.exists(segment_file):
        st.warning("customer_segments.csv not found.")
        st.info("Run customer segmentation first: python src/customer_segmentation.py")
    else:
        try:
            df = pd.read_csv(segment_file)
            if "segment" not in df.columns:
                st.error("The customer_segments.csv file does not contain a segment column.")
            else:
                st.subheader("Customer Segment Distribution")
                st.bar_chart(df["segment"].value_counts().sort_index())
                st.subheader("Segment Summary")
                if {"customer_id", "frequency", "monetary", "recency"}.issubset(df.columns):
                    summary = df.groupby("segment").agg(customers=("customer_id", "count"), avg_recency=("recency", "mean"), avg_frequency=("frequency", "mean"), avg_monetary=("monetary", "mean")).round(2)
                else:
                    summary = df.groupby("segment").size().to_frame("customers")
                st.dataframe(summary, width="stretch")
                st.subheader("Customer Segment Data")
                st.dataframe(df, width="stretch")
        except Exception as e:
            st.error(f"Customer segmentation error: {e}")
elif page == "💬 AI Shopping Assistant":
    st.header("💬 AI Shopping Assistant")
    st.write("Ask questions about products in the catalog.")
    with st.form("shopping_assistant_form"):
        question = st.text_input("Ask your question", placeholder="Example: Show me products under 1000")
        submitted = st.form_submit_button("Ask AI")
    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching products and generating answer..."):
                try:
                    result = ask_shopping_assistant(question.strip())
                    if isinstance(result, dict):
                        st.subheader("🤖 AI Answer")
                        st.write(result.get("answer", "No answer returned."))
                        if result.get("sources"):
                            st.subheader("📚 Retrieved Products")
                            st.dataframe(pd.DataFrame(result["sources"]), width="stretch")
                    elif isinstance(result, str) and result.strip():
                        st.subheader("🤖 AI Answer")
                        st.write(result)
                    else:
                        st.info("The assistant completed the request but did not return displayable text.")
                except Exception as e:
                    st.error(f"AI Assistant error: {e}")
