from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import chatbot_response, PRODUCT_DATABASE, faq_data  # ensure these exist/import correctly

app = Flask(__name__)
CORS(app)

# --- In-memory customer profiles (from the JSON you provided) ---
CUSTOMER_PROFILES = {
    "1001": {
        "purchased_products": [101, 103, 106],
        "preferences": "The customer prefers silk and cotton fabrics, and buys outfits mainly for festive occasions and parties."
    },
    "1002": {
        "purchased_products": [102, 105],
        "preferences": "The customer usually buys casual and daily wear clothes in bright colors."
    },
    "1003": {
        "purchased_products": [104],
        "preferences": "No specific preferences recorded, but only purchases expensive, high-end garments."
    }
}

@app.route("/products", methods=["GET"])
def get_products():
    """
    Returns the product database as-is.
    Frontend expects an object/dict mapping product ids -> product data (PRODUCT_DATABASE).
    """
    return jsonify(PRODUCT_DATABASE)


@app.route("/chat", methods=["POST"])
def chat():
    """
    Generic chatbot endpoint used by your existing frontend `postChat`.
    Returns JSON: {"reply": "<text>"}
    """
    data = request.get_json(silent=True) or {}
    user_query = data.get("query", "")
    try:
        answer = chatbot_response(user_query)
        return jsonify({"reply": answer})
    except Exception as e:
        app.logger.exception("chat endpoint error")
        return jsonify({"reply": "Sorry, something went wrong on the chat server."}), 500


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Recommendation endpoint. Frontend recommends using:
      POST /recommend { customer_id: <id>, query: "<text>" }
    Returns JSON: {"answer": "<text>"}
    Behavior:
      - If customer_id is present and known, the customer's preferences and purchase history
        are appended to the query before calling chatbot_response.
      - If unknown customer_id, still calls chatbot_response with original query.
    """
    data = request.get_json(silent=True) or {}
    customer_id = data.get("customer_id")
    user_query = data.get("query", "")

    # If customer id is provided and in our profiles, append profile context to the query
    if customer_id is not None:
        # ensure string keys (frontend may send numbers)
        cid_str = str(customer_id)
        profile = CUSTOMER_PROFILES.get(cid_str)
        if profile:
            # add short context for the recommender model
            prefs = profile.get("preferences", "")
            purchased = profile.get("purchased_products", [])
            purchase_str = "purchased_products: " + ", ".join(map(str, purchased)) if purchased else ""
            # Append to user query so chatbot_response can use it
            appended = f"{user_query} cid {cid_str} prefs {prefs} {purchase_str}"
            user_query = appended
        else:
            # unknown customer id - still include cid token so chatbot can handle it if needed
            user_query = f"{user_query} cid {cid_str}"

    try:
        answer = chatbot_response(user_query)
        return jsonify({"answer": answer})
    except Exception as e:
        app.logger.exception("recommend endpoint error")
        return jsonify({"answer": "Sorry, recommendation failed on the server."}), 500


# ✅ FAQ endpoint adapted for frontend FAQPanel (question + response)
@app.route("/faq", methods=["GET"])
def get_faq():
    """
    Returns list of FAQs in the shape the frontend expects:
      [{ question: "...", response: "..." }, ...]
    `faq_data` is expected to be a list of objects with keys "question" and "answer".
    """
    formatted = [
        {"question": item.get("question", ""), "response": item.get("answer", "")}
        for item in faq_data
    ]
    return jsonify(formatted)


# --- Optional: expose customers for debugging/inspection ---
@app.route("/customers", methods=["GET"])
def get_customers():
    """
    Returns the in-memory CUSTOMER_PROFILES for debugging or admin UI.
    Remove or protect this in production.
    """
    return jsonify(CUSTOMER_PROFILES)


if __name__ == "__main__":
    # run with debug for development; change as needed for prod
    app.run(debug=True, port=5000)
