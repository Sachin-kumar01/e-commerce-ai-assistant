USE ecommerce_ai;

SELECT
    c.customer_unique_id,
    COUNT(DISTINCT o.order_id) AS total_orders,
    ROUND(SUM(oi.price), 2) AS total_spending,
    ROUND(AVG(oi.price), 2) AS average_item_price
FROM customers c

JOIN orders o
    ON c.customer_id = o.customer_customer_id

JOIN order_items oi
    ON o.order_id = oi.order_id

GROUP BY c.customer_unique_id

ORDER BY total_spending DESC

LIMIT 20;