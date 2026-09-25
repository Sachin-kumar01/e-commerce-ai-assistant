# 🛍️ E-Commerce AI Assistant

An end-to-end **E-Commerce Recommendation and GenAI Shopping Assistant**
built using Python, MySQL, Machine Learning, FAISS, OpenRouter and
Streamlit.

The project combines:

-   📊 E-Commerce data analytics
-   🤖 Content-Based Recommendation
-   👥 Collaborative Filtering
-   🔀 Hybrid Recommendation
-   👤 Customer Segmentation using RFM + K-Means
-   🔎 FAISS-based product retrieval
-   💬 GenAI Shopping Assistant using RAG
-   🌐 Streamlit interactive web application

------------------------------------------------------------------------

## 📌 Project Overview

The **E-Commerce AI Assistant** is designed to simulate an intelligent
shopping platform where users can explore products, receive
recommendations, understand customer segments, and ask natural-language
questions about the product catalog.

The complete workflow is:

``` text
Indian E-Commerce Dataset
        ↓
Data Preprocessing
        ↓
MySQL Database
        ↓
SQL Analytics
        ↓
Recommendation System
   ├── Content-Based
   ├── Collaborative Filtering
   └── Hybrid Recommendation
        ↓
Customer Segmentation
        ↓
FAISS Product Knowledge Base
        ↓
RAG + OpenRouter LLM
        ↓
Streamlit AI Shopping Assistant
```

------------------------------------------------------------------------

## 🚀 Key Features

### 1. 📊 E-Commerce Dashboard

The dashboard provides high-level business metrics such as:

-   Total customers
-   Total orders
-   Total products
-   Total revenue
-   Top product categories

The dashboard retrieves data directly from MySQL.

------------------------------------------------------------------------

### 2. 🤖 Content-Based Recommendation

The content-based recommender recommends products based on product-level
attributes.

The current implementation uses:

-   Product category
-   Unit price
-   Product rating
-   TF-IDF vectorization
-   Cosine similarity

The trained model is stored locally and loaded by the Streamlit
application.

------------------------------------------------------------------------

### 3. 👥 Collaborative Filtering

The collaborative recommender uses customer-product purchase
interactions.

The workflow is:

``` text
Customer → Purchased Product
```

A sparse customer-product matrix is created to avoid the large memory
consumption associated with a dense matrix.

Product-to-product similarity is then calculated using cosine
similarity.

------------------------------------------------------------------------

### 4. 🔀 Hybrid Recommendation

The project combines Content-Based and Collaborative scores.

Current hybrid scoring:

``` text
Hybrid Score =
    0.6 × Content Score
    +
    0.4 × Collaborative Score
```

This allows the application to combine product similarity with
behavior-based recommendations.

------------------------------------------------------------------------

### 5. 👤 Customer Segmentation

Customer segmentation is implemented using **RFM analysis**:

-   Recency
-   Frequency
-   Monetary

The RFM values are transformed and standardized before applying K-Means
clustering.

The project currently uses:

``` text
Number of clusters = 4
Random state = 42
```

The segment IDs are machine-generated labels and should not be
interpreted as fixed business labels without further analysis.

------------------------------------------------------------------------

### 6. 🔎 FAISS Product Search

FAISS is used as a local product retrieval engine.

The current implementation uses:

``` text
TF-IDF
   ↓
Normalized vectors
   ↓
FAISS IndexFlatIP
```

The product knowledge base contains the processed product catalog.

The project intentionally uses a lightweight TF-IDF + FAISS approach
rather than Sentence Transformers because the local Windows environment
had PyTorch compatibility issues.

------------------------------------------------------------------------

### 7. 💬 GenAI Shopping Assistant

The AI Shopping Assistant combines product retrieval with an LLM.

Architecture:

``` text
User Question
      ↓
Smart Product Retrieval
      ↓
FAISS / Rule-Based Filters
      ↓
Retrieved Products
      ↓
Context
      ↓
OpenRouter
      ↓
AI Shopping Response
```

The assistant supports queries such as:

``` text
show me some products
```

``` text
products under 1000
```

``` text
products below 500
```

``` text
show highly rated products
```

The price and rating filters provide deterministic retrieval for common
shopping queries.

------------------------------------------------------------------------

## 🧰 Technology Stack

### Programming Language

-   Python

### Data Processing

-   Pandas
-   NumPy

### Machine Learning

-   Scikit-learn
-   TF-IDF
-   Cosine Similarity
-   K-Means Clustering

### Database

-   MySQL
-   SQLAlchemy
-   PyMySQL

### Recommendation System

-   Content-Based Filtering
-   Collaborative Filtering
-   Hybrid Recommendation

### Vector Search

-   FAISS
-   TF-IDF vectors

### Generative AI

-   OpenRouter API
-   OpenAI Python SDK

### Frontend / Application

-   Streamlit

### Environment Management

-   Python virtual environment
-   python-dotenv

### Development Tools

-   VS Code
-   Git
-   GitHub
-   MySQL Workbench

------------------------------------------------------------------------

## 📂 Project Structure

``` text
E-Commerce AI Assistant/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env
│
├── data/
│   ├── raw/
│   │   └── Ecommerce.csv
│   │
│   └── processed/
│       ├── customers.csv
│       ├── products.csv
│       ├── orders.csv
│       └── order_items.csv
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── content_similarity.pkl
│   ├── content_products.pkl
│   ├── collaborative_similarity.pkl
│   ├── collaborative_product_ids.pkl
│   ├── customer_scaler.pkl
│   ├── customer_kmeans.pkl
│   ├── customer_segments.csv
│   └── evaluation_results.csv
│
├── embeddings/
│   ├── product_faiss.index
│   ├── product_metadata.pkl
│   └── product_tfidf.pkl
│
├── notebooks/
│
├── sql_queries/
│
└── src/
    ├── __init__.py
    ├── database.py
    ├── load_data.py
    ├── preprocessing.py
    ├── recommender.py
    ├── customer_segmentation.py
    ├── evaluation.py
    ├── faiss_product_search.py
    └── rag_chatbot.py
```

> Generated model files, datasets, `.env`, and other large/local files
> should be excluded from GitHub using `.gitignore` where appropriate.

------------------------------------------------------------------------

# 📊 Dataset

The project uses the **Indian E-Commerce Customer Behavior & Purchase**
dataset.

Dataset source:

``` text
Kaggle
Indian E-Commerce Customer Behavior & Purchase
```

Dataset URL:

https://www.kaggle.com/datasets/kundanbedmutha/indian-e-commerce-customer-behavior-and-purchase

The main raw file used by the project is:

``` text
data/raw/Ecommerce.csv
```

Dataset characteristics used in this project:

``` text
Rows: 25,000
Columns: 29
```

Important columns include:

``` text
customer_id
session_id
visit_date
device_type
user_type
marketing_channel
product_id
product_category
unit_price
quantity
discount_percent
discount_amount
revenue
pages_viewed
time_on_site_sec
added_to_cart
purchased
cart_abandoned
rating
payment_method
location
```

------------------------------------------------------------------------

# 🗄️ MySQL Database Setup

Create the database:

``` sql
CREATE DATABASE ecommerce_ai;
```

Use it:

``` sql
USE ecommerce_ai;
```

The project uses four main tables:

``` text
customers
products
orders
order_items
```

------------------------------------------------------------------------

# 🔐 Environment Variables

Create a `.env` file in the project root.

``` env
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=YOUR_MYSQL_PASSWORD
MYSQL_DATABASE=ecommerce_ai

OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY
```

### ⚠️ Security

Never upload `.env` to GitHub.

Do not hard-code:

-   MySQL password
-   OpenRouter API key
-   Any private credentials

The `.gitignore` file should contain:

``` text
.env
venv/
__pycache__/
*.pyc
```

------------------------------------------------------------------------

# ⚙️ Installation

## 1. Clone the Repository

``` bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd E-Commerce-AI-Assistant
```

## 2. Create Virtual Environment

Windows:

``` powershell
python -m venv venv
```

Activate:

``` powershell
.\venv\Scripts\activate
```

------------------------------------------------------------------------

## 3. Install Dependencies

``` powershell
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🛠️ Project Setup

## Step 1 --- Configure MySQL

Make sure MySQL Server is running.

Create:

``` text
ecommerce_ai
```

database.

Update `.env` with your MySQL credentials.

------------------------------------------------------------------------

## Step 2 --- Test MySQL Connection

Run:

``` powershell
python src/database.py
```

Expected:

``` text
MySQL connected successfully!
Current database: ecommerce_ai
```

------------------------------------------------------------------------

## Step 3 --- Preprocess Dataset

Place:

``` text
Ecommerce.csv
```

inside:

``` text
data/raw/
```

Run:

``` powershell
python src/preprocessing.py
```

This creates:

``` text
data/processed/customers.csv
data/processed/products.csv
data/processed/orders.csv
data/processed/order_items.csv
```

------------------------------------------------------------------------

## Step 4 --- Load Data into MySQL

Run:

``` powershell
python src/load_data.py
```

The script loads the processed CSV files into MySQL.

Expected tables:

``` text
customers
products
orders
order_items
```

------------------------------------------------------------------------

# 🤖 Train Recommendation Models

Run:

``` powershell
python src/recommender.py
```

This prepares the recommendation models.

The project uses:

``` text
Content-Based Recommendation
Collaborative Filtering
```

The generated model files are stored inside:

``` text
models/
```

------------------------------------------------------------------------

# 👤 Train Customer Segmentation

Run:

``` powershell
python src/customer_segmentation.py
```

This creates:

``` text
models/customer_scaler.pkl
models/customer_kmeans.pkl
models/customer_segments.csv
```

------------------------------------------------------------------------

# 📈 Evaluate Recommendation System

Run:

``` powershell
python src/evaluation.py
```

The evaluation calculates:

-   Precision@5
-   Recall@5
-   Hit Rate@5

The results are saved to:

``` text
models/evaluation_results.csv
```

One evaluation run on the project data produced approximately:

  Model             Precision@5   Recall@5   Hit Rate@5
  --------------- ------------- ---------- ------------
  Content-Based          0.0014     0.0069       0.0069
  Collaborative          0.1757     0.8783       0.8783

These values are specific to the evaluation split and implementation
used in this project. They should not be interpreted as general
accuracy.

------------------------------------------------------------------------

# 🔎 Build FAISS Product Knowledge Base

Run:

``` powershell
python src/faiss_product_search.py
```

This creates:

``` text
embeddings/product_faiss.index
embeddings/product_metadata.pkl
embeddings/product_tfidf.pkl
```

The FAISS index currently contains the processed product catalog.

------------------------------------------------------------------------

# 💬 Test RAG Shopping Assistant

Make sure the OpenRouter key is present in `.env`.

Run:

``` powershell
python src/rag_chatbot.py
```

Example queries:

``` text
show me some products
```

``` text
products under 1000
```

``` text
products below 500
```

``` text
show highly rated products
```

Type:

``` text
exit
```

to stop the interactive assistant.

------------------------------------------------------------------------

# 🌐 Run Streamlit Application

Start the application:

``` powershell
streamlit run app.py
```

Open:

``` text
http://localhost:8501
```

The application contains four main sections:

``` text
📊 Dashboard
🤖 Recommendations
👥 Customer Segments
💬 AI Shopping Assistant
```

------------------------------------------------------------------------

# 🧪 Application Testing

## Dashboard

Verify:

-   Customer count
-   Order count
-   Product count
-   Revenue
-   Product category chart

------------------------------------------------------------------------

## Recommendations

Select a product and click:

``` text
Get Recommendations
```

Verify:

``` text
Content-Based Recommendations
Collaborative Recommendations
Hybrid Recommendations
```

------------------------------------------------------------------------

## Customer Segments

Verify:

``` text
Customer Segment Distribution
Segment Summary
Customer Segment Data
```

------------------------------------------------------------------------

## AI Shopping Assistant

Try:

``` text
products under 1000
```

``` text
products below 500
```

``` text
show highly rated products
```

The assistant should display:

``` text
AI Answer
Retrieved Products
```

The input is implemented using a Streamlit form, so pressing **Enter**
can submit the query.

------------------------------------------------------------------------

# 🧠 Recommendation System Details

## Content-Based Filtering

The content model converts product information into TF-IDF vectors.

Conceptually:

``` text
Product Information
       ↓
Text Representation
       ↓
TF-IDF
       ↓
Cosine Similarity
       ↓
Similar Products
```

------------------------------------------------------------------------

## Collaborative Filtering

Customer-product purchase interactions are represented using a sparse
matrix:

``` text
Customer × Product
```

Example:

``` text
        P1  P2  P3  P4
C1      1   0   2   0
C2      0   1   0   1
C3      2   0   1   0
```

Sparse representation is used to reduce memory usage.

------------------------------------------------------------------------

## Hybrid Recommendation

The current hybrid formula is:

``` text
Hybrid Score =
0.6 × Content Score
+
0.4 × Collaborative Score
```

The products are then ranked according to the combined score.

------------------------------------------------------------------------

# 👥 Customer Segmentation Details

RFM analysis is calculated using:

### Recency

How recently the customer purchased.

### Frequency

Number of unique purchase orders.

### Monetary

Total purchase revenue.

Workflow:

``` text
Purchase Data
     ↓
RFM Features
     ↓
Log Transformation
     ↓
StandardScaler
     ↓
K-Means
     ↓
4 Customer Segments
```

------------------------------------------------------------------------

# 🔎 RAG Architecture

The shopping assistant uses a retrieval-augmented generation workflow.

``` text
User Query
     ↓
Query Analysis
     ↓
Price / Rating / FAISS Retrieval
     ↓
Top Products
     ↓
Context Construction
     ↓
OpenRouter LLM
     ↓
Natural Language Answer
```

This approach allows the model to answer using retrieved product
information rather than relying only on the model's general knowledge.

------------------------------------------------------------------------

# 🧩 Why TF-IDF + FAISS?

The project initially considered transformer-based embeddings using
Sentence Transformers.

However, the local Windows environment encountered PyTorch
native-library compatibility issues.

Therefore, the final local retrieval implementation uses:

``` text
TF-IDF
+
FAISS
```

This provides a lightweight vector-search solution without requiring a
working PyTorch installation.

------------------------------------------------------------------------

# 📦 Requirements

The project uses Python packages including:

``` text
pandas
numpy
scikit-learn
sqlalchemy
pymysql
python-dotenv
joblib
faiss-cpu
openai
streamlit
```

Install all dependencies with:

``` powershell
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🔒 GitHub Security

Before pushing the project:

### Do NOT upload:

``` text
.env
venv/
__pycache__/
```

Also consider excluding generated datasets and model artifacts if they
are large.

Example `.gitignore`:

``` text
.env
venv/
__pycache__/
*.pyc
.streamlit/secrets.toml
data/raw/
data/processed/
models/*.pkl
embeddings/*.pkl
*.index
```

If you need another developer to reproduce the project, provide the
dataset source and instructions rather than committing sensitive
credentials.

------------------------------------------------------------------------

# 📌 Example User Queries

The AI Shopping Assistant can handle queries such as:

``` text
Show me some products
```

``` text
Show products under 1000
```

``` text
Products below 500
```

``` text
Show highly rated products
```

``` text
Find affordable products
```

------------------------------------------------------------------------

# 📊 Project Results

The project successfully processes:

``` text
25,000 sessions/records
8,442 customer records
899 products
4,176 customers with purchase history
```

The recommendation evaluation includes a held-out test setup.

In the recorded evaluation:

``` text
Collaborative Hit Rate@5 ≈ 87.83%
```

This means approximately 87.83% of evaluated test cases had the held-out
product appear in the top-5 recommendations under that specific
evaluation setup.

It is **not classification accuracy** and should not be described as
general model accuracy.

------------------------------------------------------------------------

# 🚀 Future Improvements

Possible future improvements include:

-   Semantic product embeddings
-   Multilingual shopping queries
-   Better product-category mappings
-   Personalized recommendations based on customer history
-   Better cold-start handling
-   Product images
-   Product availability/inventory
-   Price comparison
-   Conversational memory
-   Recommendation explanations
-   Online recommendation evaluation
-   A more advanced hybrid ranking model
-   Deployment using Streamlit Cloud or another hosting platform

------------------------------------------------------------------------

# 🎯 Resume Description

You can describe this project on your resume as:

> **E-Commerce Recommendation + GenAI Shopping Assistant** --- Built an
> end-to-end e-commerce intelligence platform using Python, MySQL,
> machine learning, FAISS and OpenRouter. Implemented content-based,
> collaborative and hybrid product recommendations, RFM-based customer
> segmentation with K-Means, and a RAG-based GenAI shopping assistant
> with a Streamlit interface.

------------------------------------------------------------------------

# 💼 Interview Explanation

A simple explanation for an interview:

> "I built an end-to-end E-Commerce AI Assistant using an Indian
> e-commerce dataset. I first processed the data and stored it in MySQL
> for analytics. Then I built content-based and collaborative
> recommendation systems and combined them using a hybrid scoring
> approach. I also performed RFM-based customer segmentation using
> K-Means. For the GenAI component, I created a FAISS product knowledge
> base and connected it with an OpenRouter LLM using a RAG pipeline.
> Finally, I integrated everything into a Streamlit dashboard where
> users can explore analytics, recommendations, customer segments and
> ask natural-language product questions."

------------------------------------------------------------------------

# 👨‍💻 Author

**Sachin Kumar**

B.Tech -- Computer Science & Engineering

Ambalika Institute of Management and Technology, Lucknow

------------------------------------------------------------------------

# ⭐ Project Highlights

``` text
✅ Indian E-Commerce Dataset
✅ MySQL Database
✅ SQL Analytics
✅ Content-Based Recommendation
✅ Collaborative Filtering
✅ Hybrid Recommendation
✅ RFM Customer Segmentation
✅ K-Means Clustering
✅ TF-IDF
✅ FAISS Vector Search
✅ RAG Architecture
✅ OpenRouter LLM
✅ Streamlit Dashboard
✅ End-to-End AI Application
```

------------------------------------------------------------------------

## 📜 License

This project is intended for educational, portfolio and demonstration
purposes.

Add an appropriate open-source license if you plan to distribute the
source code publicly.
