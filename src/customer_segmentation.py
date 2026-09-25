import os
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ============================================================
# PATHS
# ============================================================

ORDERS_FILE = "data/processed/orders.csv"
ORDER_ITEMS_FILE = "data/processed/order_items.csv"

MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

class CustomerSegmentation:

    def __init__(self):

        print("\nLoading customer order data...")

        self.orders = pd.read_csv(
            ORDERS_FILE
        )

        self.order_items = pd.read_csv(
            ORDER_ITEMS_FILE
        )

        # Convert date
        self.orders["visit_date"] = pd.to_datetime(
            self.orders["visit_date"],
            errors="coerce"
        )

        print(
            "Orders loaded:",
            len(self.orders)
        )

        print(
            "Order items loaded:",
            len(self.order_items)
        )

        # ----------------------------------------------------
        # Keep actual purchases
        # ----------------------------------------------------

        self.orders = self.orders[
            self.orders["purchased"] == 1
        ].copy()

        print(
            "Purchased orders:",
            len(self.orders)
        )


    # ========================================================
    # CREATE RFM FEATURES
    # ========================================================

    def create_rfm(self):

        print("\nCreating RFM features...")

        # ----------------------------------------------------
        # Join orders + order items
        # ----------------------------------------------------

        purchase_data = self.orders[
            [
                "order_id",
                "customer_id",
                "visit_date"
            ]
        ].merge(
            self.order_items[
                [
                    "order_id",
                    "product_id",
                    "quantity",
                    "revenue"
                ]
            ],
            on="order_id",
            how="inner"
        )

        print(
            "Purchase records:",
            len(purchase_data)
        )

        # ----------------------------------------------------
        # Reference date
        # ----------------------------------------------------

        max_date = purchase_data[
            "visit_date"
        ].max()

        reference_date = (
            max_date +
            pd.Timedelta(days=1)
        )

        # ----------------------------------------------------
        # RFM calculation
        # ----------------------------------------------------

        rfm = (
            purchase_data
            .groupby("customer_id")
            .agg(
                recency=(
                    "visit_date",
                    lambda x:
                    (
                        reference_date - x.max()
                    ).days
                ),

                frequency=(
                    "order_id",
                    "nunique"
                ),

                monetary=(
                    "revenue",
                    "sum"
                )
            )
            .reset_index()
        )

        print(
            "\nRFM customers:",
            len(rfm)
        )

        print("\nRFM sample:")

        print(
            rfm.head()
        )

        return rfm


    # ========================================================
    # K-MEANS CLUSTERING
    # ========================================================

    def create_segments(
        self,
        rfm,
        n_clusters=4
    ):

        print(
            "\nPreparing data for K-Means..."
        )

        features = [
            "recency",
            "frequency",
            "monetary"
        ]

        X = rfm[
            features
        ].copy()

        # ----------------------------------------------------
        # Log transformation
        # ----------------------------------------------------

        X["recency"] = np.log1p(
            X["recency"]
        )

        X["frequency"] = np.log1p(
            X["frequency"]
        )

        X["monetary"] = np.log1p(
            X["monetary"]
        )

        # ----------------------------------------------------
        # Standardization
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            X
        )

        print(
            "Scaled feature matrix:",
            X_scaled.shape
        )

        # ----------------------------------------------------
        # K-Means
        # ----------------------------------------------------

        print(
            "\nTraining K-Means..."
        )

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

        rfm["segment"] = (
            kmeans.fit_predict(
                X_scaled
            )
        )

        # ----------------------------------------------------
        # Save scaler
        # ----------------------------------------------------

        joblib.dump(
            scaler,
            os.path.join(
                MODELS_DIR,
                "customer_scaler.pkl"
            )
        )

        # ----------------------------------------------------
        # Save K-Means model
        # ----------------------------------------------------

        joblib.dump(
            kmeans,
            os.path.join(
                MODELS_DIR,
                "customer_kmeans.pkl"
            )
        )

        # ----------------------------------------------------
        # Save customer segments
        # ----------------------------------------------------

        rfm.to_csv(
            os.path.join(
                MODELS_DIR,
                "customer_segments.csv"
            ),
            index=False
        )

        print(
            "\nCustomer segmentation completed."
        )

        return rfm


    # ========================================================
    # SEGMENT SUMMARY
    # ========================================================

    def segment_summary(
        self,
        rfm
    ):

        print(
            "\n"
            + "=" * 60
        )

        print(
            "CUSTOMER SEGMENT SUMMARY"
        )

        print(
            "=" * 60
        )

        summary = (
            rfm
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

        print(
            summary
        )

        return summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")

    print(
        "=" * 60
    )

    print(
        "INDIAN E-COMMERCE CUSTOMER SEGMENTATION"
    )

    print(
        "=" * 60
    )

    segmentation = (
        CustomerSegmentation()
    )

    # --------------------------------------------------------
    # RFM
    # --------------------------------------------------------

    rfm = (
        segmentation.create_rfm()
    )

    # --------------------------------------------------------
    # K-Means
    # --------------------------------------------------------

    rfm = (
        segmentation.create_segments(
            rfm,
            n_clusters=4
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    segmentation.segment_summary(
        rfm
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CUSTOMER SEGMENTATION COMPLETED"
    )

    print(
        "=" * 60
    )