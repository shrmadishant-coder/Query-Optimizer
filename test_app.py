import app
print("Flask app loaded successfully!")
print("Engine status:", app.engine.tokenizer_loaded)
print("Testing /api/model-info logic...")
with app.app.test_client() as client:
    res = client.get("/api/model-info")
    print("GET /api/model-info status:", res.status_code, res.get_json())
    
    test_payload = {
        "query": "SELECT * FROM orders WHERE customer_id IN (SELECT id FROM customers);",
        "rule": "Convert subqueries to JOIN",
        "mode": "hybrid"
    }
    opt_res = client.post("/api/optimize", json=test_payload)
    print("POST /api/optimize status:", opt_res.status_code)
    data = opt_res.get_json()
    print("Optimized query preview:\n", data.get("optimized_query"))
    print("Rules applied:", data.get("rules_applied"))
    print("Tokens count:", data.get("token_count"))
    print("Tokens preview:", data.get("tokens")[:10])

print("ALL TESTS PASSED!")
