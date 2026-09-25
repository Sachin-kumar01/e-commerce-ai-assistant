USE ecommerce_ai;

SELECT
    product_id,
    COUNT(*) AS total_items_sold,
    ROUND(SUM(price), 2) AS total_revenue
FROM order_items
GROUP BY product_id
ORDER BY total_items_sold DESC
LIMIT 10;