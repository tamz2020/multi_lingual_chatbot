from sentence_transformers import SentenceTransformer, util
import torch
import requests
import langdetect
from langdetect import detect
import json
import os
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv
import re

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    # A more graceful exit is often better than a print and exit
    raise ValueError("Error: GEMINI_API_KEY not found in .env file.")

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
# The model is initialized with the client from google.generativeai,
# which governs the API call structure.
GEMINI_MODEL = genai.GenerativeModel(model_name='gemini-2.5-flash')

def load_product_database(file_name="product_database.json"):

    try:

        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)

            if isinstance(data, list):
                print(f"[SYSTEM] Successfully loaded product data from {file_name}.")
                return data
            else:
                print(f"[SYSTEM] Error: Data in {file_name} is not a list. Using empty list.")
                return []
    except FileNotFoundError:
        print(f"[SYSTEM] Error: Product database file '{file_name}' not found. Using empty list.")
        return []
    except json.JSONDecodeError:
        print(f"[SYSTEM] Error: Could not decode JSON from '{file_name}'. Using empty list.")
        return []

# --- New function to load customer history ---
def load_customer_history(file_name="customer_history.json"):
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                print(f"[SYSTEM] Successfully loaded customer history from {file_name}.")
                return data
            else:
                print(f"[SYSTEM] Error: Data in {file_name} is not a dictionary. Using empty dict.")
                return {}
    except FileNotFoundError:
        print(f"[SYSTEM] Error: Customer history file '{file_name}' not found. Recommendations will be unavailable.")
        return {}
    except json.JSONDecodeError:
        print(f"[SYSTEM] Error: Could not decode JSON from '{file_name}'. Recommendations will be unavailable.")
        return {}


PRODUCT_DATABASE = load_product_database(file_name="product_database.json")
CUSTOMER_HISTORY = load_customer_history(file_name="customer_history.json") # Load history

if not PRODUCT_DATABASE:
    print("[SYSTEM] Warning: Cannot find product data. The chatbot will only handle general FAQ.")
    #we'll let it run for general questions if no product data is found.


def update_faq():
    # General Questions
    faq_data = [
        {
            "question": "What are your business hours?",
            "response": "Our business hours are 24/7, Monday to Sunday.",
        },
        {
            "question": "How can I track my order?",
            "response": "You can track your order using the tracking link sent to your email after your purchase is shipped.",
        },
        {
            "question": "What is your return policy?",
            "response": "currently there is no return policy, but we can offer it in future. Please check our website for full details.",
        },
        {
            "question": "Do you offer international shipping?",
            "response": "No, we offer shipping only across India. Please check our website for the full list of supported locations.",
        },
        {
            "question": "How can I contact customer support?",

            "response": "You can contact our customer support team via email at support@example.com or call us at +1-234-567-890.",
        },
        {
            "question": "What payment methods do you accept?",
            "response": "We accept payments through credit/debit card, online paying methods like Google Pay or Pay TM, and cash on delivery.",
        },
        {
            "question": "Wie kann ich mein Paket verfolgen?",
            "response": "Bitte verwenden Sie die Sendungsverfolgungsnummer in Ihrer E-Mail, um den Status Ihres Pakets zu überprüfen.",
        },
        {
            "question": "what are your timings?",
            "response": "our timings are 24/7, monday to sunday",
        },
        {
            "question": "is there any return policy?",
            "response": "sorry we do not have any return policy, but we can have it in future",
        },
        {
            "question": "are you open in week ends?",
            "response": "yes, we are open 24/7, from monday to sunday",
        },
        {
            "question": "do you offer shipping?",
            "response": "yes, we offer shipping accross India",
        },
        {
            "question": "how to pay?",
            "response": "you can pay through credit/debit card, cash on delivery or by google pay or payTM",
        },
        {
            "question": "when will I recieve my order?",
            "response": "you will recieve your order with in the time limit given in your mail",
        },
        {
            "question": "how can I see the current location of my order?",
            "response": "you can track your order by using the given tracking id in your mail",
        },
        {
            "question": "where is my order?",
            "response": "you can track the current location of your order by using the tracking id provided in the mail",
        },
        {
            "question": "when my order will arrive?",
            "response": "your order will arrive with in 2days after it's shipping",
        },
        {
            "question": "how to call customer support?",
            "response": "to call customer support dial: +1234567890 or reach out through mail @example.com",
        },
        # --- hinglish () translations ---
        {
            "question": "mera order kab aayega?",
            "response": "aapka order shipping ke 2 dino ke andar aa jayega.",
        },
        {
            "question": "customer support ko contact kaise kare?",
            "response": "customer support ko contact karne ke liye dial kare: +1234567890 ya hamari email id @example.com par mail kare.",
        },
        {
            "question": "aapki timing kya hai?",
            "response": "hamari timing 24/7, somvar se ravivar hai.",
        },
        {
            "question": "payment kaise kare?",
            "response": "aap payment karne ke liye credit/debit card ya google pay/payTM ya cash on delivery ka istemal kar sakte hain.",
        },
        {
            "question": "kya aapki koi return policy hai?",
            "response": "maaf kare hamari abhi koi return policy nahi hai, par aage chal kar ho sakti hai.",
        },
        {
            "question": "mera order kaha hai?",
            "response": "aap apni mail main di gai tracking id se pata laga sakte hain ki aapka order abhi kaha hai.",
        },
        {
            "question": "koi return policy hai?",
            "response": "nahi maaf kare hamari return policy nahi hai abhi, par aage chal kar ho sakti hai.",
        },
        {
            "question": "mera order kab pahuchega?",
            "response": "aapka order shipping ke 2 dino ke andar pahuch jayega.",
        },
        {
            "question": "aapke business hours kya hain?",
            "response": "hamare business hours 24/7, somvar se ravivar hain.",
        },
        {
            "question": "kya aap international shipping karte hain?",
            "response": "nahi ham sirf India main hi shipping karte hain.",
        },
        {
            "question": "apna order track kaise kare?",
            "response": "mail main di gai tracking id ka istemal karke aap apne order ko track kar sakte hain.",
        },
        {
            "question": "main apne order ki current location kaise pata karu?",
            "response": "aap apne mail main di gai tracking id se apne order ki current locattion pata laga sakte hain.",
        },
        # --- New FAQ for Recommendations ---
        {
            "question": "Can you recommend a product for me?",
            "response": "To give you the best recommendation, I need your customer ID. Please provide your customer ID for a personalized suggestion."
        },
        {
            "question": "mujhe koi product recommend kardo.",
            "response": "aapko sabse accha recommendation dene ke liye, mujhe aapki customer ID chahiye. Kripya apni customer ID batayein."
        },
        # -----------------------------------
    ]

    # --- Dynamic Product FAQ generation ---
    dynamic_faq = []

    for product in PRODUCT_DATABASE:

        # Ensure instock_status is treated as a string for comparison
        instock_status = str(product.get("instock", 'False'))

        availability_response = (
            f"Yes, the {product['name']} is available in our stock."
            if instock_status.lower() == 'true'
            else f"The {product['name']} is currently out of stock."
        )

        description_response = product.get(
            "description", "No description available for this product."
        )

        colors_data = product.get(
            "colors", "no other colour available for this product."
        )

        if isinstance(colors_data, list):
            color_response = f"The {product['name']} is available in these colors: {', '.join(colors_data)}."
        else:
            color_response = f"The {product['name']} is available in {colors_data}."


        # Check for price presence and format it explicitly
        price_data = product.get('price')
        if price_data is None:
            price_response = 'Price information is unavailable for this product.'
        elif isinstance(price_data, (int, float)):
             # Assuming prices are in Indian Rupees (₹) for a traditional dress shop in India
             price_response = f"The price of the {product['name']} is ₹{price_data}."
        else:
             # If it's a string (e.g., '1200-1500')
             price_response = f"The price of the {product['name']} is {price_data}."


        size_response = product.get('sizes', 'No size information is available.')
        dimension_response = product.get('dimensions', 'No dimensions information is available.')
        fabric_response = product.get('fabric', 'Fabric information is unavailable.')
        occasion_response = product.get('occasion', 'Occasion information is unavailable.')
        weight_response = product.get('weight', 'Weight information is unavailable.')
        length_response = product.get('length', 'Length information is unavailable.')

        dynamic_faq.extend(
    [
        {
            "question": f"Is the {product['name']} available?",
            "response": availability_response,
        },
        {
            "question": f"Do you have the {product['name']}?",
            "response": availability_response,
        },
        {
            "question": f"What is the {product['name']} description?",
            "response": description_response,
        },
        {
            "question": f"Tell me about the {product['name']}.",
            "response": description_response,
        },
        {
            "question": f"Do you have the {product['name']} in other colours?",
            "response": color_response,
        },
        {
            "question": f"What colours are available for {product['name']}?",
            "response": color_response,
        },
        {
            "question": f"What sizes are available for the {product['name']}?",
            "response": size_response
        },
        {
            "question": f"What is the price of {product['name']}?",
            "response": price_response
        },
        {
            "question": f"What are the dimensions of {product['name']}?",
            "response": dimension_response
        },
        {
            "question" : f"what is the type of fabric in {product['name']}?",
            "response" : fabric_response
        },
        {
            "question" : f"type of fabric in {product['name']}?",
            "response" : fabric_response
        },
        {
            "question" : f"{product['name']} to be weared on which occasions?",
            "response": occasion_response
        },
        {
            "question" : f"{product['name']} is for which occasions?",
            "response": occasion_response
        },
        {
            "question" : f"what is the weight of {product['name']}?",
            "response": weight_response
        },
        {
            "question" : f"weight of {product['name']}?",
            "response": weight_response
        },
        {
            "question" : f"what is the length of {product['name']}?",
            "response": length_response
        },
        {
            "question" : f"length of {product['name']}?",
            "response": length_response
        }
    ]
)

    return faq_data + dynamic_faq

# Initial load of FAQ data and embeddings
faq_data = update_faq()
faq_questions = [faq["question"] for faq in faq_data]
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

# --- NEW HELPER FUNCTION TO PARSE PRICE RANGE ---
def parse_price_range(query: str) -> tuple[float, float]:
    """
    Parses a price range or single price from the query string.
    Returns (min_price, max_price). Returns (0.0, float('inf')) if no range is found.
    Handles 'under', 'less than', 'over', 'more than', and 'X-Y' formats.
    """
    query_lower = query.lower().replace(',', '') # Remove commas for cleaner parsing
    min_price, max_price = 0.0, float('inf')

    # Regex for X-Y or X to Y format
    range_match = re.search(r'(\d+)\s*[—\-]\s*(\d+)', query_lower)
    if range_match:
        min_price = float(range_match.group(1))
        max_price = float(range_match.group(2))
        return min(min_price, max_price), max(min_price, max_price)

    # Regex for 'under X', 'less than X', 'below X'
    under_match = re.search(r'(?:under|less than|below|upto|ke niche|se kam)\s*(\d+)', query_lower)
    if under_match:
        max_price = float(under_match.group(1))
        return min_price, max_price

    # Regex for 'over X', 'more than X', 'above X'
    over_match = re.search(r'(?:over|more than|above|se zyada)\s*(\d+)', query_lower)
    if over_match:
        min_price = float(over_match.group(1))
        return min_price, max_price

    # If a number is found near 'price' but no range/limit, assume it's a target (X-X) - simplified
    # This is a complex case; for simplicity, we'll only use explicit ranges/limits.

    return min_price, max_price

# --- MODIFIED get_rag_context function ---
def get_rag_context(user_query: str) -> tuple[str, bool]:
    """
    Finds the most relevant context.
    Returns (context_string, is_direct_faq_match)

    If is_direct_faq_match is True, the context_string is the final,
    pre-translated answer and should be returned directly.
    """
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, faq_embeddings)
    best_match_idx = torch.argmax(similarities).item()
    best_match_score = similarities[0][best_match_idx].item()
    # best_match_question = faq_data[best_match_idx]["question"] # Not strictly needed

    # --- 1. FEATURE EXTRACTION & PRODUCT DETAIL GUARD CLAUSE ---

    # Define keywords for product-specific details
    product_detail_keywords = ["price", "weight", "length", "dimensions", "colours", "fabric", "occasion", "color", "size", "available", "instock", "description", "cost", "how much", "kitne ka"]

    # Check if the query asks for product-specific details
    is_asking_for_product_detail = any(word in user_query.lower() for word in product_detail_keywords)

    # Search for a product name in the query
    product_name_found = False
    found_product = None
    for product in PRODUCT_DATABASE:
        if product["name"].lower() in user_query.lower():
            product_name_found = True
            found_product = product
            break

    # GUARD CLAUSE: If asking for a product detail without specifying a product,
    # preempt all other logic and ask for clarification.
    if is_asking_for_product_detail and not product_name_found:
        rag_context = (
            "TASK: The user is asking for a product detail (like price, weight, or color) "
            "but did not specify which product. Respond by asking the user to please "
            "specify the name of the product they are interested in. This is a crucial "
            "context cue, so the final answer should focus only on this request for "
            "clarification."
        )
        return rag_context, False # Use the LLM to translate and present this request.


    # --- 2. RECOMMENDATION LOGIC ---
    recommendation_keywords = ["recommend", "suggest", "id", "purchased", "history", "sugao", "batao", "salah do"]
    is_recommendation_query = any(word in user_query.lower() for word in recommendation_keywords)

    if is_recommendation_query and CUSTOMER_HISTORY:

        # Extract Occasion from Query
        supported_occasions = ["wedding", "party", "daily wear", "festive", "casual", "formal", "POOJA", "shaadi", "tyohar", "rozana"]
        desired_occasion = next((o for o in supported_occasions if o in user_query.lower()), None)

        # Extract Price Range
        min_price, max_price = parse_price_range(user_query)
        price_filter_str = f"Price Range: ₹{min_price} to ₹{max_price}" if min_price > 0 or max_price != float('inf') else "No specific price range."

        # Extract Customer ID (Optional for general recommendation)
        id_match = re.search(r'\b(?:cid|customer|id)?\s*(\d+)\b', user_query.lower())
        customer_id = id_match.group(1) if id_match else None


        if customer_id and customer_id not in CUSTOMER_HISTORY:
            # Direct response for missing ID, no LLM required
            return f"Customer ID {customer_id} not found in history. Please ensure the ID is correct or try a general query.", True

        # --- Product Filtering Logic ---
        purchased_product_ids = [str(pid) for pid in CUSTOMER_HISTORY.get(customer_id, {}).get('purchased_products', [])]

        # Retrieve product details for the history (only if ID is present)
        history_str = "No purchase history available."
        if customer_id:
            product_details_list = []
            for product_id in purchased_product_ids:
                product_info = next((p for p in PRODUCT_DATABASE if str(p.get('id')) == str(product_id)), None)

                if product_info:
                    product_details_list.append(
                        f"ID: {product_info.get('id')}, "
                        f"Name: {product_info.get('name')}, "
                        f"Occasion: {product_info.get('occasion', 'N/A')}"
                    )
            history_str = "; ".join(product_details_list)


        # Filtering Available Products for Recommendation
        recommendable_products_details = []

        for product in PRODUCT_DATABASE:
            # Check if available, not previously purchased
            is_available = str(product.get('instock')).lower() == 'true'
            is_new = str(product.get('id')) not in purchased_product_ids

            # Check for occasion match
            product_occasion = str(product.get('occasion', '')).lower()
            occasion_matches = (
                not desired_occasion or
                (desired_occasion.lower() in product_occasion)
            )

            # Check for price match
            product_price_raw = product.get('price')
            try:
                # Handle single price point products (e.g., 1500 or "1500")
                if isinstance(product_price_raw, (int, float, str)) and re.match(r'^\d+$', str(product_price_raw)):
                    product_price = float(product_price_raw)
                    price_matches = min_price <= product_price <= max_price
                # Handle price ranges in the product database (e.g., "1200-1500")
                elif isinstance(product_price_raw, str) and re.search(r'(\d+)\s*[—\-]\s*(\d+)', product_price_raw):
                    p_min, p_max = parse_price_range(product_price_raw)
                    # Check for any overlap between product range (p_min, p_max) and user range (min_price, max_price)
                    price_matches = max(min_price, p_min) <= min(max_price, p_max)
                else:
                    # If price is missing or unparseable, skip price filter
                    price_matches = True
            except:
                price_matches = True # Default to True on parsing error

            # Final check
            if is_available and is_new and occasion_matches and price_matches:
                recommendable_products_details.append(
                    f"{product.get('name')} (Category: {product.get('category', 'N/A')}, Occasion: {product.get('occasion', 'N/A')}, Price: ₹{product.get('price', 'N/A')})"
                )

        # Constructing the Recommendation RAG Context
        recommendable_products_str = "; ".join(recommendable_products_details)
        customer_pref_str = CUSTOMER_HISTORY.get(customer_id, {}).get('preferences', 'No specific preferences available') if customer_id else "No Customer ID provided."


        if not recommendable_products_details:
            # Fallback for no suitable products found
            fallback_message = (
                "TASK: The product search yielded no results. "
                f"FILTERS USED: Occasion: {desired_occasion if desired_occasion else 'None'}, {price_filter_str}. "
                "Kindly inform the user that no product matches ALL their criteria. Suggest they try "
                "a broader search by removing one of the filters (e.g., price or occasion)."
            )
            return fallback_message, False # Need LLM to translate the fallback nicely

        occasion_prompt = f" and the specific occasion is **{desired_occasion}**" if desired_occasion else "."

        context_str = (
            f"TASK: Provide a personalized product recommendation. "
            f"CUSTOMER HISTORY: Customer ID {customer_id} has a history of purchasing: {history_str}. "
            f"CUSTOMER PREFERENCES: {customer_pref_str}. "
            f"RECOMMENDATION CONSTRAINTS: The user is looking for a product for an occasion{occasion_prompt} with a {price_filter_str}. "
            f"AVAILABLE PRODUCTS TO SUGGEST: {recommendable_products_str}. "
            "Suggest **three** new products from the 'AVAILABLE PRODUCTS TO SUGGEST' list that best fit the customer's request. "
            "Briefly explain its description and clearly include the price in the final answer."
        )
        return context_str, False # Not a direct FAQ answer, needs LLM generation/translation

    # --- 3. HIGH-CONFIDENCE FAQ MATCH ---
    if best_match_score > 0.6:
        # High confidence match: Return the pre-translated (or English) answer directly.
        return faq_data[best_match_idx]["response"], True


    # --- 4. FALLBACK LOGIC (Low-confidence Product Lookup) ---

    rag_context = "No specific context found."

    # Only proceed to product lookup if a product name was explicitly found
    if product_name_found and found_product:
        product = found_product
        colors_data = product.get("colors", "N/A")
        colors_str = (
            ", ".join(colors_data)
            if isinstance(colors_data, list)
            else colors_data
        )

        # Building the specific RAG context
        rag_context = (
            f"Product info for '{product['name']}': "
            f"Description: {product.get('description', 'N/A')}. "
            f"Price: {product.get('price', 'N/A')}. "
            f"Sizes: {product.get('sizes', 'N/A')}. "
            f"Dimensions: {product.get('dimensions', 'N/A')}. "
            f"Availability: {'In Stock' if str(product.get('instock')).lower() == 'true' else 'Out of Stock'}. "
            f"Colors: {colors_str}. "
            f"Fabric: {product.get('fabric', 'N/A')}. "
            f"Occasion: {product.get('occasion', 'N/A')}. "
            f"Weight: {product.get('weight', 'N/A')}. "
            f"Length: {product.get('length', 'N/A')}"
        )

    # Return the RAG context. The LLM will use this to formulate an answer.
    return rag_context, False


def standardise_lang_code(lang_code: str) -> str:
    """
    Standardises the language code, mapping custom codes or correcting common issues.
    Maps 'HNGL' (Hinglish) to 'hi' (Hindi) for processing.
    """
    code_map = {
        'hngl': 'hi', # Custom mapping for Hinglish to Hindi
        'ur': 'hi' # Often mis-detected Urdu (in India/Pakistan context) to Hindi for simplicity
            }
    return code_map.get(lang_code.lower(), lang_code.lower())


def detect_user_language(text: str) -> str:
    """
    Detects the user's language, handles errors, and standardises the code.
    Returns the standardised ISO 639-1 code.
    """
    # 1. Define a length threshold for reliable detection
    # Default to English for very short or ambiguous queries.
    if len(text.strip().split()) <= 2:
        return "en"

    try:
        lang_code = detect(text)

        # 2. Use the standardisation function
        return standardise_lang_code(lang_code)

    except langdetect.LangDetectException:
        # Fallback for short or ambiguous text where detection fails (e.g., empty string)
        return "en"

'''def generate_and_translate_response(user_query: str, rag_context: str, target_lang_code: str) -> str:
    """
    Generates the RAG response in English and translates it in a single LLM call.
    Uses the globally initialized GEMINI_MODEL.
    """

    SUPPORTED_LANGS = {
        'en': 'English',
        'hi': 'Hindi (can handle Hinglish)',
        'bn': 'Bengali',
        'pa': 'Punjabi',
        'ml': 'Malayalam',
        'gu': 'Gujarati',
        'ta': 'Tamil',
    }

    target_lang_name = SUPPORTED_LANGS.get(target_lang_code, 'English') # Default to English

    #Streamlined System Instruction for Translation

    #System Instruction is now part of the prompt
    system_instruction_text = (
        "You are a friendly and knowledgeable customer support bot specializing in traditional dresses. "
        "Your primary goal is to use the provided RAG Context to generate a helpful and accurate answer to the user's query. "
        "If the RAG context is 'No specific context found.', answer based on general knowledge and your persona. "
        "Keep the answer concise and related to what the user has asked in the query."
    )

    # Translation instruction is separate and appended to the prompt for clarity
    if target_lang_code != 'en':
        translation_instruction = (
            f"\n\nFINAL TASK: Translate your complete English answer into **{target_lang_name}** (ISO code: '{target_lang_code}'). "
            f"The final output must be **ONLY** the translated text in {target_lang_name}. Do not include the English version, any introductory phrases, or notes about the translation."
        )
    else:
        translation_instruction = ""


    #Combining system instruction, RAG context, and user query into one prompt
    prompt = (
        f"INSTRUCTION: {system_instruction_text}\n\n"
        f"RAG Context:\n---\n{rag_context}\n---\n\n"
        f"User Query: {user_query}"
        f"{translation_instruction}"
    )


    try:
        # Pass the combined prompt to the LLM
        response = GEMINI_MODEL.generate_content(
            prompt
        )
        return response.text
    except Exception as e:
        print(f"[SYSTEM] Error during LLM generation: {e}")'''
def generate_and_translate_response(user_query: str, rag_context: str, target_lang_code: str) -> str:
    """
    Generates the RAG response in English and translates it in a single LLM call.
    Enforces strict domain guardrails.
    """

    SUPPORTED_LANGS = {
        'en': 'English',
        'hi': 'Hindi',
        'bn': 'Bengali',
        'pa': 'Punjabi',
        'ml': 'Malayalam',
        'gu': 'Gujarati',
        'ta': 'Tamil',
    }

    target_lang_name = SUPPORTED_LANGS.get(target_lang_code, 'English')

    # --- UPDATED PROMPT ENGINEERING START ---
    system_instruction_text = (
        "You are 'Zariyah', a polite and helpful customer support assistant for a company selling *Indian Traditional Dresses* (Sarees, Lehengas, Kurtas, etc.). "
        "\n\n"
        "*YOUR GUIDELINES:*\n"
        "1. *STRICT DOMAIN:* You generally answer questions ONLY related to our clothing products, store policies, or fashion advice regarding Indian traditional wear.\n"
        "2. *OFF-TOPIC HANDLING:* If the user asks about unrelated topics (e.g., coding, math, science, politics, general knowledge, weather), you must *POLITELY REFUSE*. "
        "   - Example Refusal: 'I specialize in traditional Indian dresses and cannot assist with [topic]. How can I help you with our collection today?'\n"
        "3. *USE CONTEXT:* Use the provided 'RAG Context' to answer specific questions about our inventory.\n"
        "4. *FALLBACK:* If the 'RAG Context' is 'No specific context found.' and the question is about dresses, give a general helpful fashion answer. If it is NOT about dresses, use the Off-Topic Handling."
    )
    

    if target_lang_code != 'en':
        translation_instruction = (
            f"\n\nFINAL TASK: Translate your response into *{target_lang_name}* (ISO code: '{target_lang_code}'). "
            f"Output *ONLY* the translated text."
        )
    else:
        translation_instruction = ""

    prompt = (
        f"INSTRUCTION: {system_instruction_text}\n\n"
        f"RAG Context:\n---\n{rag_context}\n---\n\n"
        f"User Query: {user_query}"
        f"{translation_instruction}"
    )

    try:
        response = GEMINI_MODEL.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"[SYSTEM] Error during LLM generation: {e}")
        return "Sorry, I am currently facing technical issues."

        # Fallback responses for supported languages
        error_message = "Sorry, I'm having trouble generating a response right now."
        if target_lang_code == 'hi':
            return "माफ़ कीजिए, मुझे इस समय प्रतिक्रिया देने में समस्या आ रही है।"
        elif target_lang_code in ['bn', 'pa', 'ml', 'gu', 'ta']:
            return "क्षमा करें, तकनीकी समस्या के कारण मैं अभी जवाब नहीं दे सकता।"
        else:
            return error_message

def chatbot_response(user_query):
    # Detect and standardize language
    detected_lang = detect_user_language(user_query)

    # Get context and the new direct FAQ flag
    context, is_direct_faq = get_rag_context(user_query)

    # --- FIXED LOGIC FOR DIRECT FAQ MATCH HANDLING ---
    if is_direct_faq:
        # Return the pre-translated (or English) short FAQ response directly.
        # These are pre-written/pre-translated FAQs or specific guard-clause responses
        # that should be returned as-is for speed and consistency.
        return context

    # If not a direct match (i.e., it's RAG context or "No specific context found." or a
    # 'clarification request' context), proceed to the LLM for generation/translation.
    final_response = generate_and_translate_response(
        user_query=user_query,
        rag_context=context,
        target_lang_code=detected_lang
    )
    return final_response

def main():
    print("Welcome to zariyah, I am a Customer Support Chatbot!")
    print(
        "I can answer your questions in multiple languages, including English and various Indian languages."
    )
    print("how can I help you?")
    print("Type 'exit' to end the chat.\n")
    while True:
        user_input = input("Enter your query: ")
        if user_input.lower() == "exit":
            print("zariyah: Thank you for reaching out. Have a great day!")
            break
        response = chatbot_response(user_input)
        print(f"zariyah: {response}")

if __name__ == "__main__":
    main()