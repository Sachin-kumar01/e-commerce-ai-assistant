import pandas as pd
from sqlalchemy import text

from database import engine


# ============================================================
# FILE PATHS
# ============================================================

CUSTOMERS_FILE = "data/processed/customers.csv"
PRODUCTS_FILE = "data/processed/products.csv"
ORDERS_FILE = "data/processed/orders.csv"
ORDER_ITEMS_FILE = "data/processed/order_items.csv"


# ============================================================
# LOAD CSV FILES
# ============================================================

print("Loading processed Indian e-commerce data...")

customers = pd.read_csv(CUSTOMERS_FILE)
products = pd.read_csv(PRODUCTS_FILE)
orders = pd.read_csv(ORDERS_FILE)
order_items = pd.read_csv(ORDER_ITEMS_FILE)

print("Customers:", customers.shape)
print("Products:", products.shape)
print("Orders:", orders.shape)
print("Order Items:", order_items.shape)


# ============================================================
# PREPARE DATA TYPES
# ============================================================

orders["visit_date"] = pd.to_datetime(
    orders["visit_date"],
    errors="coerce"
)


# ============================================================
# CREATE MYSQL TABLES
# ============================================================

with engine.begin() as connection:

    print("\nCreating MySQL tables...")

    connection.execute(
        text("DROP TABLE IF EXISTS order_items")
    )

    connection.execute(
        text("DROP TABLE IF EXISTS orders")
    )

    connection.execute(
        text("DROP TABLE IF EXISTS products")
    )

    connection.execute(
        text("DROP TABLE IF EXISTS customers")
    )


# ============================================================
# LOAD CUSTOMERS
# ============================================================

print("\nLoading customers...")

customers.to_sql(
    "customers",
    con=engine,
    if_exists="replace",
    index=False
)

print("Customers loaded successfully.")


# ============================================================
# LOAD PRODUCTS
# ============================================================

print("\nLoading products...")

products.to_sql(
    "products",
    con=engine,
    if_exists="replace",
    index=False
)

print("Products loaded successfully.")


# ============================================================
# LOAD ORDERS
# ============================================================

print("\nLoading orders...")

orders.to_sql(
    "orders",
    con=engine,
    if_exists="replace",
    index=False
)

print("Orders loaded successfully.")


# ============================================================
# LOAD ORDER ITEMS
# ============================================================

print("\nLoading order items...")

order_items.to_sql(
    "order_items",
    con=engine,
    if_exists="replace",
    index=False
)

print("Order items loaded successfully.")


# ============================================================
# VERIFY MYSQL
# ============================================================

print("\n======================================")
print("MYSQL DATA VERIFICATION")
print("======================================")

with engine.connect() as connection:

    for table in [
        "customers",
        "products",
        "orders",
        "order_items"
    ]:

        result = connection.execute(
            text(
                f"SELECT COUNT(*) FROM {table}"
            )
        )

        count = result.fetchone()[0]

        print(
            f"{table}: {count} rows"
        )


print("\nIndian e-commerce data loaded into MySQL successfully!")