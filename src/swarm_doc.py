import os
import logging
from swarms import Agent, GroupChat, round_robin_speaker
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm
from swarms import MixtureOfAgents
from swarms.structs.concurrent_workflow import ConcurrentWorkflow
import sys
import os
from src.utils import markdown_to_html

# Add the parent directory of src/ to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

agentic_qa = None

def set_agentic_rag(rag):
    """Called from app.py to inject the advanced RAG."""
    global agentic_qa
    agentic_qa = rag
    # print("[swarm2] Injected RAG detected:", type(agentic_qa))
    # DEBUG: Print available methods to console
    # print("[swarm2] Available attributes:", dir(agentic_qa))


from dotenv import load_dotenv

# ---------------------------
# Configuration & globals
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DOCUMENT_SUMMARY = None


load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")
os.environ["GEMINI_API_KEY"] = gemini_api_key


model = "gemini/gemini-2.5-flash"

def create_group_chat():
    """
    Fast Multi-Agent GroupChat for Scenario 3.
    3 swarm agents, but only 1 round of discussion.
    Uses RAG_CONTEXT as the shared evidence source.
    """

    logger.info("Creating Optimized Group Chat...")

    shared_prompt = (
        "You have only one turn in this discussion. Provide your full reasoning and complete role-specific contribution in a single, comprehensive response.")    

    agents = [
        Agent(
            agent_name="Senior-Physician",
            system_prompt=(
                "You are a senior attending physician. Analyze the case, "
                "identify the key issues, and guide the team toward the final diagnosis."
            ),
            model_name=model,
            max_loops=1,
        ),
        Agent(
            agent_name="Resident-Doctor",
            system_prompt=(
                "You are a medical resident. Explain the RAG_CONTEXT, present the findings, "
                "and suggest early differential diagnoses."
            ),
            model_name=model,
            max_loops=1,
        ),
        Agent(
            agent_name="Pharmacist",
            system_prompt=(
                "You are a clinical pharmacist. Review medication suitability, interactions, "
                "and provide dosing recommendations using the RAG_CONTEXT."
            ),
            model_name=model,
            max_loops=1,
        ),
    ]

    group_chat = GroupChat(
        agents=agents,
        description="Medical swarm: Senior physician, resident doctor, and pharmacist collaborating.",
        max_loops=1,              
        speaker_fn=round_robin_speaker
    )

    return group_chat


def create_summarizer_agent():
    return Agent(
        agent_name="Final-Summarizer",
        system_prompt=(
            "You are a medical summarization specialist. "
            "Given the COMPLETE transcript of a multi-doctor discussion, "
            "produce a concise, structured medical summary for Malaysian citizens in layman terms including:\n"
            "- Key findings\n"
            "- Differential diagnosis\n"
            "- Agreed diagnosis\n"
            "- Treatment plan\n"
            "- Risks & follow-up\n"
            "Do NOT add new facts. Summarize only what was discussed."
        ),
        model_name="gemini/gemini-2.5-flash",
        max_loops=1
    )
def generate_document_summary(document_text: str):
    global DOCUMENT_SUMMARY
    if DOCUMENT_SUMMARY:
        return DOCUMENT_SUMMARY  # return cached summary
    
    summarizer = Agent(
        agent_name="Document-Summarizer",
        system_prompt=(
            "You are a medical document summarization expert. "
            "Summarize the ENTIRE document into structured medical data:\n"
            "- Symptoms\n"
            "- Medical history\n"
            "- Vitals\n"
            "- Lab results\n"
            "- Risk factors\n"
            "- Timeline of events\n"
            "Be concise but complete. Do NOT add external knowledge."
        ),
        model_name=model,
        max_loops=1,
    )

    summary = summarizer.run(document_text)
    DOCUMENT_SUMMARY = summary  # cache it
    
    return summary
    
def create_query_expansion_agent():
    return Agent(
        agent_name="Query-Expansion",
        system_prompt=(
            "You rewrite vague user queries into retrieval-friendly queries for RAG.\n\n"
            "Inputs you will receive:\n"
            "1. User's raw query\n"
            "2. Summary of the uploaded medical document\n\n"
            "Output ONLY the optimized search query. No explanations."
        ),
        model_name=model,
        max_loops=1,
    )
def expand_query(user_query: str):
    summary = DOCUMENT_SUMMARY  # already generated earlier
    
    expander = create_query_expansion_agent()
    
    refined_query = expander.run(
        f"USER QUERY:\n{user_query}\n\n"
        f"DOCUMENT SUMMARY:\n{summary}\n\n"
        "Rewrite this into an improved retrieval query."
    )
    
    return refined_query
    
def rag_once(user_query: str, chat_history: list, document_text: str):
    """Run: generate summary once -> expand query -> call AgenticQA RAG pipeline once -> return context/answer."""
    # 1) generate and cache document summary (only first time)
    generate_document_summary(document_text)

    # 2) expand the user query using the document summary
    optimized_query = expand_query(user_query)
    logger.info(f"Expanded query for RAG: {optimized_query}")


    # 3) call AgenticQA's RAG pipeline with expanded_query (AgenticQA must accept expanded_query)
    # Note: AgenticQA._run_rag_with_reranking signature expected as: (query, chat_history, expanded_query=None)
    rag_context = agentic_qa.run_rag_with_reranking(query=user_query, chat_history=chat_history, expanded_query=optimized_query,return_only_context=True)

    return rag_context

# ============================================================================
# MAIN EXECUTION - TEST ALL TECHNIQUES
# ============================================================================
def run_medical_swarm(document_text, user_query):
    gc = create_group_chat()
    rag_context = rag_once(user_query, chat_history=[], document_text=document_text)
    print(rag_context)
    # --- shared single-turn instruction ---
    shared_prompt = (
        "You have only one turn in this discussion. "
        "Provide your full reasoning and complete role-specific contribution in a single, comprehensive response. "
        "Use the provided RAG_CONTEXT when it contains relevant information: explicitly cite any facts taken from it under a 'Sources used:' section. "
        "If the RAG_CONTEXT contradicts your internal knowledge, prioritize the RAG_CONTEXT and explain the discrepancy briefly. "
        "If the RAG_CONTEXT is not relevant, state 'No relevant context found' and answer concisely from your medical knowledge."
    )
    prompt = (
        f"--- CURRENT PATIENT DATA (Analyze THIS person) ---\n"
        f"{document_text}\n\n"
        
        f"--- PATIENT'S QUESTION ---\n"
        f"{user_query}\n\n"
        
        f"--- REFERENCE KNOWLEDGE (For context/comparison only) ---\n"
        f"The following is a similar case retrieved from the medical database. "
        f"DO NOT confuse this with the current patient:\n"
        f"{rag_context}\n\n"
        
        f"--- INSTRUCTIONS FOR ALL PARTICIPANTS ---\n"
        f"{shared_prompt}\n\n"
    )
    discussion = gc.run(prompt)
    summarizer = create_summarizer_agent()
    final_summary = summarizer.run(
        f"Transcript:\n{discussion}\n\nSummarize the final medical conclusion."
    )
    final_summary=markdown_to_html(final_summary)
    return final_summary

# if __name__ == "__main__":
#     sample_document = """
#     PATIENT: Gelap Benau
#     AGE: 45
#     SYMPTOMS: Persistent headaches, dizziness, and blurred vision for 2 weeks. 
#     BP: 150/75 mmHg. HR: 80 bpm. 
#     HISTORY: Type 2 Diabetes (diagnosed 2022), Smoker (3 pack/week).
#     """
    
#     sample_query = "What is wrong with me and what should I do?"
    
#     print("\n" + "="*80)
#     print("TESTING ADVANCED SWARM TECHNIQUES")
#     print("="*80 + "\n")
    
#     # ===== TECHNIQUE 1: MIXTURE OF AGENTS =====
#     print("\n" + "-"*80)
#     print("TECHNIQUE 1: MIXTURE OF AGENTS (Multiple Experts + Aggregator)")
#     print("-"*80)
#     # try:
#     #     moa = create_mixture_of_agents()
#     #     task = f"{sample_query}\n\n{sample_document}"
#     #     result = moa.run(task)
#     #     print("\n✅ MoA Result:")
#     #     print(result)
#     # except Exception as e:
#     #     logger.error(f"MoA failed: {e}")
#     #     import traceback
#     #     traceback.print_exc()
    
#     # # ===== TECHNIQUE 2: SPREADSHEET SWARM =====
#     # print("\n" + "-"*80)
#     # print("TECHNIQUE 2: SPREADSHEET SWARM (Concurrent + CSV Logging)")
#     # print("-"*80)
#     # try:
#     #     spreadsheet = create_spreadsheet_swarm()
#     #     task = f"{sample_query}\n\n{sample_document}"
#     #     result = spreadsheet.run(task)
#     #     print("\n✅ SpreadSheet Swarm Result:")
#     #     print(result)
#     #     print("📄 Results saved to: medical_analysis_results.csv")
#     # except Exception as e:
#     #     logger.error(f"SpreadSheet Swarm failed: {e}")
#     #     import traceback
#     #     traceback.print_exc()
    
#     # # ===== TECHNIQUE 3: CONCURRENT WORKFLOW =====
#     # print("\n" + "-"*80)
#     # print("TECHNIQUE 3: CONCURRENT WORKFLOW (Specialist Consultation)")
#     # print("-"*80)
#     # try:
#     #     concurrent = create_concurrent_workflow()
#     #     task = f"{sample_query}\n\n{sample_document}"
#     #     result = concurrent.run(task)
#     #     print("\n✅ Concurrent Workflow Results:")
#     #     if isinstance(result, dict):
#     #         for agent_name, output in result.items():
#     #             print(f"\n--- {agent_name} ---")
#     #             print(output)
#     #     else:
#     #         print(result)
#     # except Exception as e:
#     #     logger.error(f"Concurrent Workflow failed: {e}")
#     #     import traceback
#     #     traceback.print_exc()
    
    # ===== TECHNIQUE 4: GROUP CHAT =====
    # print("\n" + "-"*80)
    # print("TECHNIQUE 4: GROUP CHAT (Team Discussion)")
    # print("-"*80)
    # try:
    #     chat = create_group_chat()
    #     # 2) Run single-shot RAG to get context for the group chat
    #     rag_result = rag_once(sample_query, chat_history=[], document_text=sample_document)
    #     print("✅ RAG result (used as shared context for GroupChat):")
    #     print(rag_result)

    #     # 3) Feed the RAG result + user query into the GroupChat
    #     task = f"User question: {sample_query}  Context for discussion: {rag_result} "
    #     "Please discuss and come to a consensus."
    #     discussion = chat.run(task)
    #     print("✅ Group Chat Discussion:")
    #     print(discussion)


    #     # 4) Summarize the discussion
    #     summarizer = create_summarizer_agent()
    #     summary = summarizer.run(f"Here is the full transcript of the team discussion: {discussion}"
    #     "Please summarize the team conclusions.")
    #     print("=== FINAL SUMMARY ===")
    #     print(summary)
    # except Exception as e:
    #     logger.error(f"Group Chat failed: {e}")
    #     import traceback
    #     traceback.print_exc()
    
#     print("\n" + "="*80)
#     print("ALL TECHNIQUES TESTED")
#     print("="*80)