USE ecommerce_ai;

SELECT
    a.product_id AS product_a,
    b.product_id AS product_b,
    COUNT(*) AS times_bought_together

FROM order_items a

JOIN order_items b
    ON a.order_id = b.order_id
    AND a.product_id < b.product_id

GROUP BY
    a.product_id,
    b.product_id

ORDER BY times_bought_together DESC

LIMIT 20;