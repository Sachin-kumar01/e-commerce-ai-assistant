import pandas as pd
import os

# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_FILE = "data/raw/Ecommerce.csv"

OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading Indian e-commerce dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)


# ============================================================
# 3. BASIC CLEANING
# ============================================================

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing important values
important_columns = [
    "customer_id",
    "session_id",
    "visit_date",
    "product_id",
    "product_category",
    "unit_price",
    "quantity",
    "revenue"
]

df = df.dropna(
    subset=important_columns
)


# ============================================================
# 4. DATE CONVERSION
# ============================================================

df["visit_date"] = pd.to_datetime(
    df["visit_date"],
    format="%d-%m-%Y",
    errors="coerce"
)

# Remove invalid dates
df = df.dropna(
    subset=["visit_date"]
)


# ============================================================
# 5. CREATE ORDER ID
# ============================================================

# Dataset has session_id instead of traditional order_id.
# We will use session_id as the transaction/order identifier.

df["order_id"] = (
    "ORD_" +
    df["session_id"].astype(str)
)


# ============================================================
# 6. CREATE CUSTOMER TABLE
# ============================================================

customers = df[
    [
        "customer_id",
        "location",
        "device_type",
        "user_type"
    ]
].drop_duplicates(
    subset=["customer_id"]
)


# ============================================================
# 7. CREATE PRODUCT TABLE
# ============================================================

products = df[
    [
        "product_id",
        "product_category",
        "unit_price",
        "rating"
    ]
].drop_duplicates(
    subset=["product_id"]
)


# ============================================================
# 8. CREATE ORDERS TABLE
# ============================================================

orders = df[
    [
        "order_id",
        "customer_id",
        "session_id",
        "visit_date",
        "device_type",
        "marketing_channel",
        "payment_method",
        "purchased",
        "cart_abandoned",
        "pages_viewed",
        "time_on_site_sec"
    ]
].drop_duplicates(
    subset=["order_id"]
)


# ============================================================
# 9. CREATE ORDER ITEMS TABLE
# ============================================================

order_items = df[
    [
        "order_id",
        "product_id",
        "quantity",
        "unit_price",
        "discount_percent",
        "discount_amount",
        "revenue",
        "rating"
    ]
].copy()


# ============================================================
# 10. SAVE PROCESSED FILES
# ============================================================

customers.to_csv(
    f"{OUTPUT_DIR}/customers.csv",
    index=False
)

products.to_csv(
    f"{OUTPUT_DIR}/products.csv",
    index=False
)

orders.to_csv(
    f"{OUTPUT_DIR}/orders.csv",
    index=False
)

order_items.to_csv(
    f"{OUTPUT_DIR}/order_items.csv",
    index=False
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n======================================")
print("PREPROCESSING COMPLETED")
print("======================================")

print("\nCustomers:")
print(customers.shape)

print("\nProducts:")
print(products.shape)

print("\nOrders:")
print(orders.shape)

print("\nOrder Items:")
print(order_items.shape)


print("\nFiles created:")

print("data/processed/customers.csv")
print("data/processed/products.csv")
print("data/processed/orders.csv")
print("data/processed/order_items.csv")