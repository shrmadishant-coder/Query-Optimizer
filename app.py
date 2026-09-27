import os
import sys
import time
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from model import QueryOptimizerEngine
from optimizer_rules import SQLRuleOptimizer

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize engine
print("Initializing Query Optimizer Engine...")
engine = QueryOptimizerEngine(base_dir=BASE_DIR)
print(f"Engine ready. Tokenizer: {engine.tokenizer_loaded}, Weights loaded: {engine.weights_loaded}, Device: {engine.device}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    return jsonify({
        "status": "online",
        "architecture": "Transformer Decoder-Only (GPT)",
        "layers": engine.num_layers,
        "attention_heads": engine.num_heads,
        "embedding_dim": engine.d_model,
        "feedforward_dim": engine.d_ff,
        "vocab_size": engine.vocab_size,
        "device": str(engine.device),
        "tokenizer_loaded": engine.tokenizer_loaded,
        "weights_loaded": engine.weights_loaded,
        "checkpoint": engine.checkpoint_name,
        "total_parameters": 4934007
    })


@app.route("/api/tokenize", methods=["POST"])
def tokenize():
    data = request.get_json() or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Empty text provided"}), 400

    token_info = engine.tokenize_text(text)
    return jsonify(token_info)


@app.route("/api/format", methods=["POST"])
def format_sql():
    data = request.get_json() or {}
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    formatted = SQLRuleOptimizer.format_sql(query)
    return jsonify({"formatted_query": formatted})


@app.route("/api/optimize", methods=["POST"])
def optimize_query():
    start_time = time.time()
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    rule = data.get("rule", "").strip()
    engine_mode = data.get("mode", "hybrid")  # hybrid, neural, rules
    temperature = float(data.get("temperature", 0.7))
    max_tokens = int(data.get("max_tokens", 80))

    if not query:
        return jsonify({"error": "Please provide an SQL query to optimize."}), 400

    # 1. Rule-based evaluation
    rule_results = SQLRuleOptimizer.optimize(query, rule=rule)

    # 2. Tokenize original query with SentencePiece model
    token_data = engine.tokenize_text(query)

    # 3. Neural inference
    neural_results = None
    if engine_mode in ["neural", "hybrid"]:
        neural_results = engine.generate_optimized_query(
            prompt=query,
            rule=rule,
            max_tokens=max_tokens,
            temperature=temperature
        )

    # Determine final query response depending on mode and weights availability
    final_optimized_query = rule_results["optimized_query"]
    model_note = ""

    if engine_mode == "neural":
        if neural_results and neural_results.get("raw_output"):
            final_optimized_query = neural_results["raw_output"]
            model_note = "Generated directly by the Neural Transformer Decoder."
        else:
            final_optimized_query = rule_results["optimized_query"]
            model_note = "Neural model output fallback to rule analysis (weights checkpoint pending)."
    elif engine_mode == "hybrid":
        if neural_results and engine.weights_loaded and neural_results.get("raw_output"):
            final_optimized_query = neural_results["raw_output"]
            model_note = "Synthesized using Neural Transformer model with AST validation."
        else:
            final_optimized_query = rule_results["optimized_query"]
            model_note = "Optimized via Rule-Based Analyzer with SentencePiece token mapping."
    else:  # rules only
        final_optimized_query = rule_results["optimized_query"]
        model_note = "Optimized using deterministic SQL Pattern Rules."

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return jsonify({
        "success": True,
        "original_query": query,
        "optimized_query": final_optimized_query,
        "formatted_query": rule_results["formatted_query"],
        "rules_applied": rule_results["rules_applied"],
        "explanations": rule_results["explanations"],
        "tips": rule_results["tips"],
        "risk_level": rule_results["risk_level"],
        "model_note": model_note,
        "tokens": token_data["tokens"],
        "token_ids": token_data["token_ids"],
        "token_count": token_data["length"],
        "elapsed_ms": elapsed_ms,
        "engine_mode": engine_mode,
        "neural_raw": neural_results.get("raw_output") if neural_results else None
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 Server starting on http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
