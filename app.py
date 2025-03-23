# app.py
from flask import Flask, request, jsonify, render_template
import os
import requests
import json
import base64
import numpy as np
from dotenv import load_dotenv
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

app = Flask(__name__)

# API keys
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# Preloaded list of GIF variations
AVATAR_GIFS = [
    "static/gifs/talking1.gif",
    "static/gifs/talking2.gif",
    "static/gifs/thinking.gif",
    "static/gifs/listening.gif",
]

# Load and initialize the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load company knowledge base
def load_company_kb():
    # Check if KB exists, otherwise create sample data
    kb_path = "knowledge_base.csv"

    if os.path.exists(kb_path):
        kb_df = pd.read_csv(kb_path)
    else:
        # Sample data to start with
        kb_data = {
            "question": [
                "How do I reset my password?",
                "What are the company holidays?",
                "How do I submit expense reports?",
                "How do I connect to the office WiFi?",
                "What is the VPN address?",
                "How do I book a meeting room?",
                "What's the dress code policy?",
                "How do I request time off?",
                "How do I contact IT support?",
                "What's the company's return to office policy?",
            ],
            "answer": [
                "To reset your password, visit the account settings page and click on 'Forgot Password'. Follow the instructions sent to your email.",
                "Company holidays include New Year's Day, Memorial Day, Independence Day, Labor Day, Thanksgiving, and Christmas. Please check the HR portal for the complete list.",
                "Submit expense reports through the Finance Portal with receipts attached. Reports must be submitted within 30 days of purchase.",
                "Connect to 'Company-WiFi' network using the password posted in the break room or available from your team lead.",
                "The VPN address is vpn.company.com. Use your regular network credentials to log in.",
                "Book meeting rooms through the Outlook calendar by selecting 'Rooms' when creating a new meeting invitation.",
                "Our dress code is business casual Monday through Thursday, and casual on Friday. No shorts or flip-flops please.",
                "Request time off through the HR portal at least two weeks in advance. Emergency requests should be discussed with your manager.",
                "Contact IT support at support@company.com or call extension 4357 (HELP). For urgent issues, use the emergency IT hotline.",
                "Currently, we have a hybrid work policy requiring in-office presence Tuesday through Thursday, with Monday and Friday as optional remote days.",
            ],
        }
        kb_df = pd.DataFrame(kb_data)

        # Create embeddings for the sample data
        kb_df["embedding"] = kb_df["question"].apply(lambda x: model.encode(x))

        # Save to CSV for future use (serialize numpy arrays to strings)
        kb_df_save = kb_df.copy()
        kb_df_save["embedding"] = kb_df_save["embedding"].apply(
            lambda x: json.dumps(x.tolist())
        )
        kb_df_save.to_csv(kb_path, index=False)

    # If loading from file, convert string embeddings back to numpy arrays
    if "embedding" in kb_df.columns and isinstance(kb_df["embedding"].iloc[0], str):
        kb_df["embedding"] = kb_df["embedding"].apply(lambda x: np.array(json.loads(x)))

    return kb_df


# Load knowledge base at startup
kb_df = load_company_kb()


def search_vector_db(query, threshold=0.65):
    """Search the vector database for relevant information using semantic similarity"""
    # Encode the query
    query_embedding = model.encode(query)

    # Calculate cosine similarity with all KB entries
    kb_df["similarity"] = kb_df["embedding"].apply(
        lambda x: cosine_similarity([query_embedding], [x])[0][0]
    )

    # Get the best match
    best_match = kb_df.loc[kb_df["similarity"].idxmax()]

    # Return the answer if the similarity is above threshold
    if best_match["similarity"] >= threshold:
        return best_match["answer"], True, best_match["similarity"]

    return None, False, 0.0


def get_llm_response(query, context=None):
    """Get response from Mistral API with optional context from KB"""
    try:
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        }

        system_message = "You are a helpful assistant for a help desk. Keep responses concise and informative."

        # If context is provided, enhance the system message
        if context:
            system_message = f"""You are a helpful assistant for a company help desk. 
I found some relevant information that might help answer the query.
Relevant information: {context}

Use this information to answer the query if applicable. If the information doesn't fully address the query, 
you can provide additional general guidance. Keep responses concise and informative."""

        data = {
            "model": "mistral-tiny",
            "messages": [
                {
                    "role": "system",
                    "content": system_message,
                },
                {"role": "user", "content": query},
            ],
            "max_tokens": 150,
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"LLM API Error: {e}")
        return "I'm sorry, I couldn't process that request."


def text_to_speech(text):
    """Convert text to speech using ElevenLabs"""
    try:
        url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY,
        }
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
        }

        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode("utf-8")
        else:
            print(f"TTS Error: {response.text}")
            return None
    except Exception as e:
        print(f"TTS Request Error: {e}")
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_text", methods=["POST"])
def process_text():
    data = request.json
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # First check vector knowledge base
    kb_response, found, confidence = search_vector_db(user_query)

    if found:
        response_text = kb_response
        response_source = "knowledge_base"
        confidence_score = round(confidence * 100)
    else:
        # Try to find similar but not exact matches to provide context
        partial_match, _, confidence = search_vector_db(user_query, threshold=0.5)

        # Fall back to LLM, with context if available
        response_text = get_llm_response(user_query, context=partial_match)
        response_source = "llm"
        confidence_score = 0

        if partial_match:
            response_source = "llm_with_context"
            confidence_score = round(confidence * 100)

    # Get audio data if available
    audio_data = text_to_speech(response_text)

    # Select talking GIF
    import random

    avatar_gif = random.choice(AVATAR_GIFS[:2])  # Use only talking GIFs

    return jsonify(
        {
            "text_response": response_text,
            "audio_data": audio_data,
            "avatar_gif": avatar_gif,
            "source": response_source,
            "confidence": confidence_score,
        }
    )


@app.route("/process_audio", methods=["POST"])
def process_audio():
    data = request.json
    transcribed_text = data.get("transcribed_text", "")

    if not transcribed_text:
        return jsonify({"error": "No transcribed text provided"}), 400

    # Same logic as process_text route
    kb_response, found, confidence = search_vector_db(transcribed_text)

    if found:
        response_text = kb_response
        response_source = "knowledge_base"
        confidence_score = round(confidence * 100)
    else:
        partial_match, _, confidence = search_vector_db(transcribed_text, threshold=0.5)
        response_text = get_llm_response(transcribed_text, context=partial_match)
        response_source = "llm"
        confidence_score = 0

        if partial_match:
            response_source = "llm_with_context"
            confidence_score = round(confidence * 100)

    # Get audio data if available
    audio_data = text_to_speech(response_text)

    # Select talking GIF
    import random

    avatar_gif = random.choice(AVATAR_GIFS[:2])  # Use only talking GIFs

    return jsonify(
        {
            "text_response": response_text,
            "audio_data": audio_data,
            "avatar_gif": avatar_gif,
            "source": response_source,
            "confidence": confidence_score,
        }
    )


@app.route("/manage_kb", methods=["GET"])
def manage_kb_page():
    return render_template("manage_kb.html")


@app.route("/api/knowledge", methods=["GET"])
def get_knowledge():
    global kb_df
    # Return all KB entries
    kb_list = kb_df[["question", "answer"]].to_dict(orient="records")
    return jsonify({"items": kb_list})


@app.route("/api/knowledge", methods=["POST"])
def add_knowledge():
    global kb_df

    data = request.json
    question = data.get("question", "").strip()
    answer = data.get("answer", "").strip()

    if not question or not answer:
        return jsonify({"error": "Question and answer are required"}), 400

    # Create embedding for new entry
    embedding = model.encode(question)

    # Add to dataframe
    new_row = pd.DataFrame(
        {"question": [question], "answer": [answer], "embedding": [embedding]}
    )

    kb_df = pd.concat([kb_df, new_row], ignore_index=True)

    # Save updated KB
    kb_save = kb_df.copy()
    kb_save["embedding"] = kb_save["embedding"].apply(lambda x: json.dumps(x.tolist()))
    kb_save.to_csv("knowledge_base.csv", index=False)

    return jsonify({"success": True, "id": len(kb_df) - 1})


@app.route("/api/knowledge/<int:item_id>", methods=["DELETE"])
def delete_knowledge(item_id):
    global kb_df

    if item_id < 0 or item_id >= len(kb_df):
        return jsonify({"error": "Item not found"}), 404

    # Remove the entry
    kb_df = kb_df.drop(item_id).reset_index(drop=True)

    # Save updated KB
    kb_save = kb_df.copy()
    kb_save["embedding"] = kb_save["embedding"].apply(lambda x: json.dumps(x.tolist()))
    kb_save.to_csv("knowledge_base.csv", index=False)

    return jsonify({"success": True})


if __name__ == "__main__":
    # Create directory for GIFs if it doesn't exist
    os.makedirs("static/gifs", exist_ok=True)
    app.run(debug=True)
