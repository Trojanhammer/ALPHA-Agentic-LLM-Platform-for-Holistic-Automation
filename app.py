

from flask import Flask, request, render_template, session, redirect, url_for
from flask_session import Session
from dotenv import load_dotenv
import logging
import traceback
import base64
import time
import os

# -------------------------
# New Multi-Agent Swarm
# -------------------------
from src.swarm_qa import MixtureOfAgentsQA

# -------------------------
# Document Swarm (Scenario 3)
# -------------------------
from src.swarm_doc import run_medical_swarm, set_agentic_rag
from src.utils import load_rag_system

# -------------------------
# Force logging globally so swarm_test logs show
# -------------------------
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY missing")

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
Session(app)

logger.info("🚀 Starting Multi-Domain Swarm Web App...")

# ====================================================================
# INITIALIZE SWARMS
# ====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DB = os.path.join(SCRIPT_DIR, "chroma_db_swarm")

# NEW multi-agent swarms
medical_qa = MixtureOfAgentsQA(domain="medical", persist_directory=PERSIST_DB)
islamic_qa = MixtureOfAgentsQA(domain="islamic", persist_directory=PERSIST_DB)
insurance_qa = MixtureOfAgentsQA(domain="insurance", persist_directory=PERSIST_DB)

rag2_systems = {
    'medical': load_rag_system(collection_name="medical_csv_Agentic_retrieval", domain="medical")}

# Connect new RAG to old swarm2 (document scenario)
set_agentic_rag(rag2_systems['medical'])

rag_systems = {
    "medical": medical_qa,
    "islamic": islamic_qa,
    "insurance": insurance_qa
}

logger.info("✅ All QA engines initialized.")


# ====================================================================
# HOME
# ====================================================================
@app.route("/")
def homePage():
    session.clear()
    session.pop('medical_history', None)
    session.pop('islamic_history', None)
    session.pop('insurance_history', None)
    return render_template("homePage.html")


# ====================================================================
# MEDICAL PAGE
# Supports:
#   S1: Query
#   S2: Query + Image
#   S3: Query + Document (swarm2)
# ====================================================================
@app.route("/medical", methods=["GET", "POST"])
def medical_page():
    if request.method == "GET":
        latest = session.pop("latest_medical_response", {})
        return render_template(
            "medical_page.html",
            history=session.get("medical_history", []),
            answer=latest.get("answer", ""),
            source=latest.get("source", ""),
            thoughts=latest.get("thoughts", ""),
            validation=latest.get("validation", "")
        )

    # POST
    answer = ""
    source = ""
    validation = ""
    thoughts = ""
    history = session.get("medical_history", [])
    current_doc = session.get("current_medical_document", "")

    try:
        query = request.form.get("query", "").strip()
        file_img = request.files.get("image")
        file_doc = request.files.get("document")

        has_image = file_img and file_img.filename
        has_doc = file_doc and file_doc.filename
        start_time = time.time()
        # ---------------------------------------------------------------
        # Scenario 3 — Query + Document (swarm2)
        # ---------------------------------------------------------------
        if has_doc:
            logger.info("📘 Scenario 3: Query + Document (swarm2)")
            doc_text = file_doc.read().decode("utf-8", errors="ignore")
            session["current_medical_document"] = doc_text  # store for follow-ups

            swarm_answer = run_medical_swarm(doc_text, query)

            answer = swarm_answer
            source = "swarm2 (Document Pipeline)"
            validation = True
            thoughts = "Document pipeline executed."

            # Store in history (raw dict form)
            history.append({"role": "user", "content": f"[Document Uploaded] {query}"})
            history.append({"role": "assistant", "content": answer})

        # ---------------------------------------------------------------
        # Scenario 2 — Query + Image (MixtureOfAgentsQA)
        # ---------------------------------------------------------------
        elif has_image:
            logger.info("🖼 Scenario 2: Query + Image")

            upload_dir = "Uploads"
            os.makedirs(upload_dir, exist_ok=True)
            image_path = os.path.join(upload_dir, file_img.filename)
            file_img.save(image_path)

            agent = rag_systems["medical"]
            response_dict = agent.answer(
                query=query,
                chat_history=history,
                image_path=image_path
            )

            answer = response_dict["answer"]
            source = response_dict["source"]
            validation = True
            thoughts = "Image processed successfully."

            # update history from swarm_test
            history = response_dict["chat_history"]

            os.remove(image_path)

        # ---------------------------------------------------------------
        # Scenario 1 — Query Only (MixtureOfAgentsQA)
        # ---------------------------------------------------------------
        elif query:
            logger.info("💬 Scenario 1: Query Only")

            agent = rag_systems["medical"]
            response_dict = agent.answer(
                query=query,
                chat_history=history
            )
            # ADD THIS DEBUG PRINT
            print(f"\n🔍 DEBUG - Response dict keys: {response_dict.keys()}")
            print(f"🔍 DEBUG - Has 'answer' key? {'answer' in response_dict}")
            print(f"🔍 DEBUG - Answer value type: {type(response_dict.get('answer'))}")
            print(f"🔍 DEBUG - Answer length: {len(str(response_dict.get('answer', '')))}")
            print(f"🔍 DEBUG - Chat history type: {type(response_dict.get('chat_history'))}")
            print(f"🔍 DEBUG - Chat history length: {len(response_dict.get('chat_history', []))}")
            # Print the last message in chat history
            chat_history = response_dict.get('chat_history', [])
            if chat_history:
                print(f"🔍 DEBUG - Last message in history: {chat_history[-1]}")
                print(f"🔍 DEBUG - Last message content length: {len(chat_history[-1].get('content', ''))}")
            answer = response_dict["answer"]
            source = response_dict["source"]
            validation = True
            thoughts = "Query processed."

            # update history from swarm_test
            history = response_dict["chat_history"]

        else:
            raise ValueError("No query provided.")
        
        latency = time.time() - start_time
        print(f"⏱️ QUERY LATENCY Medical ) = {latency:.3f} seconds")

    except Exception as e:
        logger.error("❌ Error in /medical", exc_info=True)
        answer = f"Error: {e}"
        thoughts = traceback.format_exc()
    
    # Save
    session["medical_history"] = history
    session["latest_medical_response"] = {
        "answer": answer,
        "source": source,
        "thoughts": thoughts,
        "validation": validation
    }
    session.modified = True

    return redirect(url_for("medical_page"))


@app.route("/medical/clear")
def clear_medical_chat():
    session.pop("medical_history", None)
    session.pop("current_medical_document", None)
    logger.info("Medical chat history cleared.")
    return redirect(url_for("medical_page"))


# ====================================================================
# ISLAMIC PAGE
# ====================================================================
@app.route("/islamic", methods=["GET", "POST"])
def islamic_page():
    if request.method == "GET":
        latest = session.pop("latest_islamic_response", {})
        return render_template(
            "islamic_page.html",
            history=session.get("islamic_history", []),
            answer=latest.get("answer", ""),
            source=latest.get("source", ""),
            thoughts=latest.get("thoughts", ""),
            validation=latest.get("validation", "")
        )

    answer = ""
    thoughts = ""
    source = ""
    validation = ""
    history = session.get("islamic_history", [])

    try:
        query = request.form.get("query", "").strip()
        file_img = request.files.get("image")
        has_img = file_img and file_img.filename

        agent = rag_systems["islamic"]
        start_time = time.time()
        if has_img:
            logger.info("🕌 Islamic: Query + Image")

            upload_dir = "Uploads"
            os.makedirs(upload_dir, exist_ok=True)
            image_path = os.path.join(upload_dir, file_img.filename)
            file_img.save(image_path)

            response_dict = agent.answer(
                query=query,
                chat_history=history,
                image_path=image_path
            )

            os.remove(image_path)

        else:
            logger.info("🕌 Islamic: Query Only")
            response_dict = agent.answer(query=query, chat_history=history)
            
        latency = time.time() - start_time
        logger.info(f"⏱️ QUERY LATENCY Islamic ) = {latency:.3f} seconds")
        answer = response_dict["answer"]
        source = response_dict["source"]
        history = response_dict["chat_history"]
        validation = True

        
    except Exception as e:
        logger.error("❌ Error in /islamic", exc_info=True)
        answer = f"Error: {e}"
        thoughts = traceback.format_exc()

    session["islamic_history"] = history
    session["latest_islamic_response"] = {
        "answer": answer,
        "source": source,
        "thoughts": thoughts,
        "validation": validation
    }
    session.modified = True

    return redirect(url_for("islamic_page"))


@app.route("/islamic/clear")
def clear_islamic_chat():
    session.pop("islamic_history", None)
    logger.info("Islamic chat history cleared.")
    return redirect(url_for("islamic_page"))


# ====================================================================
# INSURANCE PAGE
# ====================================================================
@app.route("/insurance", methods=["GET", "POST"])
def insurance_page():
    if request.method == "GET":
        latest = session.pop("latest_insurance_response", {})
        return render_template(
            "insurance_page.html",
            history=session.get("insurance_history", []),
            answer=latest.get("answer", ""),
            source=latest.get("source", ""),
            thoughts=latest.get("thoughts", ""),
            validation=latest.get("validation", "")
        )
    answer = ""
    thoughts = ""
    source = ""
    validation = ""
    history = session.get("insurance_history", [])

    try:
        query = request.form.get("query", "").strip()
        if not query:
            raise ValueError("No query provided.")

        agent = rag_systems["insurance"]
        start_time = time.time()
        response_dict = agent.answer(query=query, chat_history=history)
        latency = time.time() - start_time
        print(f"⏱️ QUERY LATENCY Insurance ) = {latency:.3f} seconds")
        answer = response_dict["answer"]
        source = response_dict["source"]
        history = response_dict["chat_history"]
        validation = True

    except Exception as e:
        logger.error("❌ Error in /insurance", exc_info=True)
        answer = f"Error: {e}"
        thoughts = traceback.format_exc()

    session["insurance_history"] = history
    session["latest_insurance_response"] = {
        "answer": answer,
        "source": source,
        "thoughts": thoughts,
        "validation": validation
    }
    session.modified = True

    return redirect(url_for("insurance_page"))


@app.route("/insurance/clear")
def clear_insurance_chat():
    session.pop("insurance_history", None)
    logger.info("Insurance chat history cleared.")
    return redirect(url_for("insurance_page"))

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")

# ====================================================================
# START
# ====================================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
