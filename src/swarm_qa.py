"""
Optimized Multi-Domain Swarms System
Key Changes:
1. Strategic RAG - Only where it adds value over LLM knowledge
2. Shared contexts - Medical agents use same RAG results
3. Tool composition - Agents share tools intelligently
4. Reasoning mode - Some agents rely on LLM knowledge when appropriate
"""
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Literal
from swarms import Agent, MixtureOfAgents
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from flashrank import Ranker, RerankRequest
from typing import List, Dict, Optional
import logging
import json
import re
import os
from langchain_community.tools.tavily_search import TavilySearchResults
import base64
from langchain_core.messages import HumanMessage
from datetime import datetime
import time
import csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# from src.utils import markdown_to_html
import re

def extract_from_bold_answer(text: str) -> str:
    """
    Extracts everything from the first bold **Direct Answer** onwards.
    If not found, falls back to bold **Detailed Explanation**.
    If none found, returns the original text.
    """

    # Strict pattern: bold Direct Answer only
    # Matches: **Direct Answer**, **Direct Answer:**, ** Direct Answer **
    direct_pattern = re.compile(
        r"\*\*\s*direct answer\s*[:]*\s*\*\*",
        re.IGNORECASE
    )

    detailed_pattern = re.compile(
        r"\*\*\s*detailed explanation\s*[:]*\s*\*\*",
        re.IGNORECASE
    )

    # Try Direct Answer first
    match = direct_pattern.search(text)
    if match:
        return text[match.start():].strip()

    # Fallback: Detailed Explanation
    match = detailed_pattern.search(text)
    if match:
        return text[match.start():].strip()

    # No match: return full text
    return text.strip()

def stop_when_no_answer_generated(output):
    """
    Stop the agent if output contains "No answer generated"
    Returns True to stop, False to continue
    """
    # Convert output to string
    output_str = str(output).strip()
    
    # Check for the phrase (case-insensitive)
    if "none" in output_str.lower():
        return True  # Stop the agent
    
    return False  # Continue

class ParallelSwarmExecutor:
    """Executes multiple agents in parallel with timeout control"""
    
    def __init__(self, timeout_per_agent: int = 30):
        self.timeout_per_agent = timeout_per_agent
        
    def execute_parallel(self, agents: List, query: str) -> Dict[str, str]:
        """
        Execute all agents in parallel and collect their responses
        
        Args:
            agents: List of Agent instances
            query: The query to process
            
        Returns:
            Dict with agent_name: response pairs
        """
        results = {}
        
        def run_agent(agent):
            """Run a single agent with timeout"""
            try:
                # Clear agent memory before each run
                if hasattr(agent, 'short_memory'):
                    agent.short_memory.clear()
                
                # Run the agent
                return agent.run(query)
            except Exception as e:
                logger.error(f"Agent {agent.agent_name} failed: {e}")
                return f"Error: {str(e)}"
        
        # Execute all agents in parallel
        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            # Submit all tasks
            future_to_agent = {
                executor.submit(run_agent, agent): agent 
                for agent in agents
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    result = future.result(timeout=self.timeout_per_agent)
                    results[agent.agent_name] = result
                except TimeoutError:
                    logger.warning(f"Agent {agent.agent_name} timed out")
                    results[agent.agent_name] = "Timeout - no response"
                except Exception as e:
                    logger.error(f"Agent {agent.agent_name} error: {e}")
                    results[agent.agent_name] = f"Error: {str(e)}"
        
        return results

# ============================================================================
# SHARED RAG PIPELINE (Only for collections that matter)
# ============================================================================

class SpecializedRAGPipeline:
    """RAG pipeline - now only used where it adds value"""
    
    def __init__(self, collection_name: str, persist_directory: str, domain: str, specialty: str):
        self.domain = domain
        self.specialty = specialty
        self.collection_name = collection_name
        
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
        
        self.db_client = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_function,
            collection_name=self.collection_name
        )
        
        self.retriever = self.db_client.as_retriever()
        
        # Load docs for BM25
        docs_for_bm25 = []
        try:
            all_docs_data = self.db_client.get()
            if all_docs_data and 'documents' in all_docs_data and len(all_docs_data['documents']) > 0:
                docs_for_bm25 = [
                    Document(page_content=content, metadata=meta)
                    for content, meta in zip(
                        all_docs_data['documents'], 
                        all_docs_data['metadatas'] if all_docs_data['metadatas'] else [{}] * len(all_docs_data['documents'])
                    )
                ]
                print(f"Loaded {len(docs_for_bm25)} docs for {specialty}")
        except Exception as e:
            logger.error(f"Error loading docs: {e}")
        
        self.hybrid_retriever = self._create_hybrid_retriever(docs_for_bm25)
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        
    def _create_hybrid_retriever(self, documents):
        class MiniSwarmRetriever:
            def __init__(self, dense_retriever, docs):
                self.dense_retriever = dense_retriever
                self.bm25_retriever = None
                
                if docs and len(docs) > 0:
                    try:
                        self.bm25_retriever = BM25Retriever.from_documents(docs)
                        self.bm25_retriever.k = 20
                    except Exception as e:
                        logger.error(f"BM25 init failed: {e}")
            
            def retrieve(self, query: str, k: int) -> list:
                retrieval_tasks = {
                    "dense": lambda: self.dense_retriever.invoke(query, k=k)
                }
                
                if self.bm25_retriever:
                    retrieval_tasks["bm25"] = lambda: self.bm25_retriever.invoke(query)[:k]
                
                all_docs = []
                seen = set()
                
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = {executor.submit(task): name for name, task in retrieval_tasks.items()}
                    
                    for future in as_completed(futures):
                        try:
                            for doc in future.result():
                                content_hash = hash(doc.page_content[:200])
                                if content_hash not in seen:
                                    seen.add(content_hash)
                                    all_docs.append(doc)
                        except Exception as e:
                            logger.error(f"Retrieval error: {e}")
                
                return all_docs
        
        return MiniSwarmRetriever(self.retriever, documents)
    
    def retrieve_and_rerank(self, query: str, k: int = 10, threshold: float = 0.1) -> List[Document]:
        swarm_docs = self.hybrid_retriever.retrieve(query, k=k)
        
        if not swarm_docs:
            return []
        
        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata} 
            for i, doc in enumerate(swarm_docs)
        ]
        
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = self.reranker.rerank(rerank_request)
        
        final_docs = []
        for result in reranked_results:
            if result['score'] > threshold:
                doc = Document(
                    page_content=result['text'],
                    metadata={**result.get('meta', {}), 'specialty': self.specialty}
                )
                final_docs.append(doc)
        
        return final_docs
    
class SwarmRunner:
    """
    Wrapper around Swarms MixtureOfAgents (MOA) that adds:
    - Early stopping
    - Interpretability trace logs
    - Timing per loop
    - Clean final answer return
    """

    def __init__(self, moa, max_loops=3, enable_trace=True):
        """
        Parameters:
        - moa: Your MixtureOfAgents instance
        - max_loops: Maximum number of swarm iterations
        - enable_trace: Enables interpretability logging
        """
        self.moa = moa
        self.max_loops = max_loops
        self.enable_trace = enable_trace

        # Stores logs for research/visualization
        self.research_trace = []

    # ---------------------------------------------------------
    # ðŸ‘‡ MAIN EXECUTION WRAPPER
    # ---------------------------------------------------------
    def run(self, query):
        self.research_trace = []
        final_answer = None

        for loop_index in range(self.max_loops):

            start_time = time.time()

            # -------------------------------------------------
            # ðŸ”¥ CALL THE INTERNAL SWARM ENGINE
            # -------------------------------------------------
            step_output = self.moa.run(query=query)

            elapsed = round(time.time() - start_time, 3)

            # -------------------------------------------------
            # ðŸ” PARSE TOOL-CALL INFORMATION
            # -------------------------------------------------
            function_call = getattr(step_output, "function_call", None)
            tool_calls = getattr(step_output, "tool_calls", None)

            is_tool_call = bool(function_call or tool_calls)

            # -------------------------------------------------
            # ðŸ“˜ Add to RESEARCH TRACE (for UI/analysis)
            # -------------------------------------------------
            if self.enable_trace:
                trace_entry = {
                    "loop": loop_index,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "latency": elapsed,
                    "raw_output": str(step_output),
                    "is_tool_call": is_tool_call
                }

                if function_call:
                    trace_entry["tool_name"] = function_call.name
                    trace_entry["tool_args"] = function_call.arguments

                if tool_calls:
                    trace_entry["multiple_tool_calls"] = tool_calls

                self.research_trace.append(trace_entry)

            # -------------------------------------------------
            # ðŸ›‘ EARLY STOPPING
            # -------------------------------------------------
            if not is_tool_call:
                # Means final answer
                final_answer = step_output
                break

            # If tool-call: Swarm internally handles tool execution
            # and retries on the next loop.

        # If all loops exhausted but no final answer:
        if final_answer is None:
            final_answer = step_output

        return final_answer, self.research_trace
    
# ============================================================================
# SHARED TOOL FACTORY
# ============================================================================

class SharedToolFactory:
    """Creates reusable tools that multiple agents can share"""
    
    @staticmethod
    def create_rag_tool(rag_pipeline, parent_qa_system, default_k=5):
        """RAG search tool factory"""
        def rag_search(query: str) -> str:
            """
            Search internal database for relevant documents.
            Use this FIRST before trying web search.
            
            Args:
                query (str): The search query to find relevant documents
                
            Returns:
                str: Retrieved context from database or error message
            """
            try:
                k = getattr(parent_qa_system, 'current_k', default_k)
                docs = rag_pipeline.retrieve_and_rerank(query=query, k=k, threshold=0.10)
                
                if docs and len(docs) > 0:
                    llm_context = "\n\n".join([d.page_content for d in docs])
                      # Build context for evaluation
                    eval_context = {
                        "query": query,
                        "doc_count": len(docs),
                        "timestamp": datetime.now().isoformat(),
                        "docs": [
                            {
                                "content": d.page_content,
                                "metadata": getattr(d, "metadata", {})
                            }
                            for d in docs
                        ]
                    }
                     # Store for evaluation
                    if hasattr(parent_qa_system, 'eval_logs'):
                        parent_qa_system.eval_logs.append(eval_context)
                    
                    return f"[RAG: {len(docs)} documents]\n{llm_context}"
                
                else:
                    return "[RAG: No documents found]"
            except Exception as e:
                logger.error(f"RAG error: {e}")
                return "[RAG Error]"
        
        rag_search.__name__ = "RAG_Search"
        return rag_search
    
    @staticmethod
    def create_domain_web_search(domain: str, sites: List[str]):
        """Domain-specific web search factory"""
        search_engine = TavilySearchResults(max_results=3)
        
        def domain_search(query: str) -> str:
            """
            Search specific authoritative websites for domain-specific information.
            
            Args:
                query (str): The search query
                
            Returns:
                str: Search results from domain-specific websites
            """
            try:
                site_filter = " OR ".join([f"site:{s}" for s in sites])
                search_query = f"{site_filter} {query}"
                results = search_engine.invoke(search_query)
                return f"[Domain Web Search]\n{str(results)}"
            except Exception as e:
                return f"[Search Error: {e}]"
        
        domain_search.__name__ = f"{domain.title()}_Web_Search"
        return domain_search
    
    @staticmethod
    def create_general_web_search():
        """General web search factory"""
        search_engine = TavilySearchResults(max_results=2)
        
        def general_search(query: str) -> str:
            """
            Search the general web for broad information.
            Use this as fallback when specialized searches fail.
            
            Args:
                query (str): The search query
                
            Returns:
                str: General web search results
            """
            try:
                results = search_engine.invoke(query)
                return f"[General Web Search]\n{str(results)}"
            except Exception as e:
                return f"[Search Error: {e}]"
        
        general_search.__name__ = "General_Web_Search"
        return general_search
    
class SimpleLangChainManager:
    """Stateless manager - fixes memory bleeding"""
    
    def __init__(self, domain: str, expert_list: List[str]):
        self.domain = domain
        self.expert_list = expert_list
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1
        )
        
        self.routing_guidance = self._get_routing_guidance()
        
        # Build system prompt BEFORE creating template
        system_prompt = self._build_system_prompt()
        
        # Create prompt template with pre-built system prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{query}")
        ])
        
        self.chain = self.prompt | self.llm
        logger.info(f"âœ… LangChain Manager for {domain}")
    
    def _get_routing_guidance(self) -> str:
        """Domain-specific routing rules"""
        guides = {
                    "islamic": """
        ROUTING RULES:
        - Single madhab question -> that expert only
        - General fiqh (if no madhab mentioned in query) -> shafii , maliki 
        - All madhabs -> shafii, maliki, hanafi_hanbali
        - Hadith -> hadith
        - History/places -> general_islamic
        """,
                    "medical": """
        ROUTING RULES:
        - Symptoms -> gp
        - Medication -> pharmacist
        - Diet/nutrition -> nutrition
        - Lifestyle + symptoms -> gp, nutrition
        - Complex cases -> multiple experts
        """,
                    "insurance": """
        ROUTING RULES:
        - Etiqa questions -> etiqa
        - Other companies -> general_insurance
        - No company mentioned -> etiqa (default)
        - Etiqa + other company -> etiqa & general_insurance
        """
        }
        return guides.get(self.domain, "Route based on query content")
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with NO template variables"""
        
        example_json = '{{"complexity": "simple", "experts": ["expert1"], "reasoning": "one sentence of reason here"}}'
        
        prompt = f"""You are a routing manager for the {self.domain} domain.

        AVAILABLE EXPERTS: {self.expert_list}

        {self.routing_guidance}

        COMPLEXITY LEVELS:
        - simple: Single expert, straightforward question
        - moderate: 2-3 experts or some depth/ one expert with complex question
        - complex: Multiple experts, multi-faceted

        YOUR TASK:
        Analyze the query and return a JSON object with these fields:
        1. complexity: one of "simple", "moderate", or "complex"
        2. experts: array of expert names from the available list
        3. reasoning: one brief sentence explaining your choice

        EXAMPLE OUTPUT:
        {example_json}

        CRITICAL RULES:
        - Return ONLY the JSON object
        - NO markdown code blocks
        - NO extra text before or after JSON
        - Experts must be from the available list
        - Always include at least one expert
        """
        return prompt
    
    def route(self, query: str) -> dict:
        """Route a query to appropriate experts"""
        try:
            # Invoke chain
            response = self.chain.invoke({"query": query})
            text = response.content if hasattr(response, 'content') else str(response)
            print("\n===== MANAGER RAW OUTPUT =====")
            print(response.content)
            print("================================\n")
            # Parse JSON from response
            result = self._extract_json(text)
            
            # Validate and fix
            result = self._validate_result(result)
            
            logger.info(f"ðŸ“Š Routed to {result['experts']} (complexity: {result['complexity']})")
            return result
            
        except Exception as e:
            logger.error(f"âŒ Routing error: {e}")
            return self._fallback_routing()
    
    def _extract_json(self, text: str) -> dict:
        """Extract JSON from LLM response"""
        # Remove markdown blocks
        text = text.replace("```json", "").replace("```", "").strip()
        
        # Find JSON objects
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        if not matches:
            raise ValueError(f"No JSON found in response: {text[:200]}")
        
        # Parse the last match
        return json.loads(matches[-1])
    
    def _validate_result(self, result: dict) -> dict:
        """Validate and fix routing result"""
        # Ensure required fields exist
        if "experts" not in result:
            result["experts"] = [self.expert_list[0]]
        
        if "complexity" not in result:
            result["complexity"] = "moderate"
        
        if "reasoning" not in result:
            result["reasoning"] = "Routing decision"
        
        # Ensure experts is a list
        if not isinstance(result["experts"], list):
            result["experts"] = [result["experts"]]
        
        # Filter to valid experts only
        valid_experts = [e for e in result["experts"] if e in self.expert_list]
        
        if not valid_experts:
            logger.warning(f"âš ï¸ Invalid experts {result['experts']}, using default")
            valid_experts = [self.expert_list[0]]
        
        result["experts"] = valid_experts
        
        return result
    
    def _fallback_routing(self) -> dict:
        """Fallback routing when parsing fails"""
        return {
            "complexity": "moderate",
            "experts": [self.expert_list[0]] if self.expert_list else ["general"],
            "reasoning": "Fallback routing due to parsing error"
        }

# ============================================================================
# OPTIMIZED EXPERT FACTORY
# ============================================================================

class OptimizedExpertFactory:
    """Creates agents with strategic tool assignments"""
    
    def __init__(self, base_llm, persist_directory: str = "./chroma_db_swarm"):
        self.base_llm = base_llm
        self.persist_directory = persist_directory
        self.expert_agents = {}
        self.rag_pipelines: Dict[str, SpecializedRAGPipeline] = {}
        self.parent_qa_system = None
        
        # Initialize shared tools
        self.tool_factory = SharedToolFactory()
        self.general_web = self.tool_factory.create_general_web_search()
        self.islamic_web = self.tool_factory.create_domain_web_search(
            "islamic", 
            ["muftiwp.gov.my", "zulkiflialbakri.com","IslamQA.org"]
        )
        self.etiqa_web = self.tool_factory.create_domain_web_search(
            "etiqa",
            ["etiqa.com.my"]
        )
        self.medical_web = self.tool_factory.create_domain_web_search(
            "medical",
            ["nhs.uk", "mayoclinic.org", "moh.gov.my", "myhealth.gov.my"]
        )
    
    def create_islamic_experts(self) -> Dict[str, Agent]:
        """Islamic domain - strategic RAG usage"""
        model="gemini/gemini-2.5-flash"
        # 1. Shafi'i Expert (HAS RAG + Web)
        shafii_rag = SpecializedRAGPipeline(
            collection_name="shafii_books",
            persist_directory=self.persist_directory,
            domain="islamic",
            specialty="shafii"
        )
        self.rag_pipelines['shafii'] = shafii_rag
        shafii_rag_tool = self.tool_factory.create_rag_tool(shafii_rag, self.parent_qa_system)
        def simple_stop(output):
            return "No response generated" in str(output).lower()
        shafii_agent = Agent(
            agent_name="Shafii_Expert",
            agent_description=(
                "Expert in Shafi'i madhab fiqh. You have access to Umdat al-Salik, "
                "Minhaj al-Talibin, and other Shafi'i texts. Provide rulings and "
                "evidence from Shafi'i perspective with proper citations."
            ),
            system_prompt=(
                "You are a Shafi'i fiqh expert.\n\n"
                "**Tools Available:**\n"
                "1. RAG_Search - Search Shafi'i texts (Umdat al-Salik, etc.)\n"
                "2. Islamic_Web_Search - Search Malaysian Islamic websites\n"
                "3. General_Web_Search - Broad web search\n\n"
                "**Workflow:**\n"
                "1. ALWAYS try RAG_Search first for fiqh rulings\n"
                "2. If RAG fails ,do Islamic_Web_Search\n"
                "3. If still unclear ,do General_Web_Search\n"
                "4. Cite sources explicitly\n"
            ),
            model_name=model,
            max_loops=3,
            verbose=False,
            react_on=False,
            tools=[shafii_rag_tool, self.islamic_web, self.general_web],
            stopping_func=stop_when_no_answer_generated,
            dynamic_temperature_enabled=False,
            temperature=0.1,
            tool_retry_attempts=1,
            tool_call_summary=True
        )
        
        # 2. Maliki Expert (HAS RAG + Web)
        maliki_rag = SpecializedRAGPipeline(
            collection_name="maliki_books",
            persist_directory=self.persist_directory,
            domain="islamic",
            specialty="maliki"
        )
        self.rag_pipelines['maliki'] = maliki_rag
        maliki_rag_tool = self.tool_factory.create_rag_tool(maliki_rag, self.parent_qa_system)
        
        maliki_agent = Agent(
            agent_name="Maliki_Expert",
            agent_description=(
                "Expert in Maliki madhab fiqh. You have access to Muwatta Imam Malik "
                "and other Maliki texts. Provide rulings from Maliki perspective."
            ),
            system_prompt=(
                "You are a Maliki fiqh expert.\n\n"
                "**Tools:** RAG_Search , Islamic_Web_Search , General_Web_Search\n"
                "Follow the same workflow as Shafi'i expert.\n"
            ),
            model_name=model,
            max_loops=3,
            react_on=False,
            verbose=False,
            tools=[maliki_rag_tool, self.islamic_web, self.general_web],
            dynamic_temperature_enabled=False,
            stopping_func=stop_when_no_answer_generated,
            temperature=0.1,
            tool_retry_attempts=1,
            tool_call_summary=True
        )
        
        # 3. Hanafi & Hanbali Expert (NO RAG - relies on LLM knowledge + Web)
        hanafi_hanbali_agent = Agent(
            agent_name="Hanafi_Hanbali_Expert",
            agent_description="Expert in Hanbali and Hanafi madhabs.",
            system_prompt=(
                "You are an expert in Hanafi and Hanbali madhabs.\n\n"
                "**Tools:** General_Web_Search only\n\n"
                "**Strategy:**\n"
                "1. Use your training knowledge of Hanafi/Hanbali fiqh\n"
                "2. Search web for verification or recent fatwas\n"
                "3. Be transparent: 'Based on Hanafi principles...' or 'According to [web source]'\n"
            ),
            model_name=model,
            max_loops=2,
            react_on=False,
            verbose=False,
            tools=[self.general_web],
            dynamic_temperature_enabled=False,
            temperature=0.1,
            stopping_func=stop_when_no_answer_generated,
            tool_retry_attempts=1,
            tool_call_summary=True
        )
        
        # 4. Hadith Specialist (NO RAG - Gemini knows hadiths!)
        hadith_agent = Agent(
            agent_name="Hadith_Specialist",
            agent_description=(
                "Expert in hadith narrations from major collections. "
                "Provides authentic narrations with proper citations."
            ),
            system_prompt=(
                "You are a hadith expert with comprehensive knowledge of:\n"
                "- Sahih Bukhari\n"
                "- Sahih Muslim\n"
                "- Sunan Abu Dawud\n"
                "- Jami' at-Tirmidhi\n"
                "- Sunan an-Nasa'i\n"
                "- Sunan Ibn Majah\n\n"
                "When asked about hadiths:\n"
                "1. Provide the hadith text in English\n"
                "2. Include the narrator chain if known\n"
                "3. Cite the collection, book, and hadith number\n"
                "4. Mention the authenticity grade (Sahih, Hasan, Da'if)\n"
                "5. Provide brief context or explanation\n\n"
                "Format: '[Collection, Book X, Hadith Y] - (Grade): Narration text'\n"
                "Example: '[Sahih Bukhari, Book 2, Hadith 7] - (Sahih): Abu Huraira reported...'"
            ),
            model_name=model,
            max_loops=1,
            react_on=False, 
            verbose=False,
            tools=None,
            dynamic_temperature_enabled=False,
            temperature=0.1, 
            stopping_func=stop_when_no_answer_generated,
        )
        
        # 5. General Islamic Expert (NO RAG - knowledge + web)
        general_islamic_agent = Agent(
            agent_name="General_Islamic_Expert",
            agent_description=(
                "Expert in general Islamic knowledge including history, figures, concepts, and places"
            ),
            system_prompt=(
                "You are an Islamic general knowledge expert.\n"
                "Use the tool General_Web_Search first.If no relevant context is found, use your internal knowledge but admit whether you use internal knowledge or websearch\n\n"
            ),
            model_name="gemini/gemini-2.5-flash",
            max_loops=2,
            verbose=False,
            react_on=False,
            tools=[self.general_web],
            dynamic_temperature_enabled=False,
            temperature=0.1,
            stopping_func=stop_when_no_answer_generated,
            tool_retry_attempts=1,
            tool_call_summary=True
        )
        
        self.expert_agents.update({
            'shafii': shafii_agent,
            'maliki': maliki_agent,
            'hanafi_hanbali': hanafi_hanbali_agent,
            'hadith': hadith_agent,
            'general_islamic': general_islamic_agent
        })
        
        logger.info("âœ… Islamic experts created ")
        return self.expert_agents
    
    def create_medical_experts(self) -> Dict[str, Agent]:
        """Medical domain - IMPROVED with specialization + safety"""
    
        # Shared medical RAG (keep this)
        medical_rag = SpecializedRAGPipeline(
            collection_name="medical_general",
            persist_directory=self.persist_directory,
            domain="medical",
            specialty="medical"
        )
        self.rag_pipelines['medical_shared'] = medical_rag
        medical_rag_tool = self.tool_factory.create_rag_tool(medical_rag, self.parent_qa_system)
        
        medical_web = self.tool_factory.create_domain_web_search(
            "medical",
            ["mayoclinic.org", "nhs.uk", "moh.gov.my", "myhealth.gov.my"]
        )
    
        
        # ðŸ©º 2. GP AGENT - Enhanced with symptom analysis
        gp_agent = Agent(
            agent_name="GP_Agent",
            agent_description="General Practitioner specializing in symptom assessment",
            system_prompt=(
                "You are a General Practitioner AI assistant.\n\n"
                "**YOUR ROLE:**\n"
                "- Assess symptoms and suggest possible causes (differential diagnosis)\n"
                "- Recommend appropriate care level (self-care, GP visit, specialist)\n"
                "- Explain when to seek immediate care\n\n"
                "**WORKFLOW:**\n"
                "1. Use RAG_Search for clinical guidelines\n"
                "2. Use Medical_Web_Search if no relevant document found and for current medical consensus\n"
                "3. Consider lifestyle factors (sleep, diet, stress)\n"
                "4. Provide actionable advice\n\n"
                "**SAFETY RULES:**\n"
                "- NEVER diagnose definitively - use 'may indicate', 'could be'\n"
                "- ALWAYS recommend professional consultation for persistent symptoms\n"
                "- Flag red flags immediately\n"
                "- Avoid dosing recommendations (defer to pharmacist)\n\n"
                "**OUTPUT FORMAT:**\n"
                "1. Symptom Summary\n"
                "2. Possible Causes (most to least likely)\n"
                "3. Self-Care Recommendations\n"
                "4. When to See a Doctor\n"
                "5. Red Flags to Watch For\n"
            ),
            model_name="gemini/gemini-2.5-flash",  # Upgraded for better reasoning
            max_loops=4,
            react_on=False,
            verbose=False,
            tools=[medical_rag_tool, medical_web, self.general_web],
            temperature=0.3,
            dynamic_temperature_enabled=True,
            stopping_func=stop_when_no_answer_generated,
            tool_retry_attempts=1,
            tool_call_summary=True
        )
        
        # ðŸ’Š 3. PHARMACIST AGENT - Medication-focused
        pharmacist_agent = Agent(
            agent_name="Pharmacist_Agent",
            agent_description="Pharmacist specializing in medications and drug interactions",
            system_prompt=(
                "You are a licensed Pharmacist AI assistant.\n\n"
                "**SPECIALIZATION:**\n"
                "- Drug interactions and contraindications\n"
                "- Medication side effects and warnings\n"
                "- Over-the-counter vs prescription guidance\n"
                "- Proper medication usage and storage\n\n"
                "**WORKFLOW:**\n"
                "1. RAG_Search for drug information databases\n"
                "2. Medical_Web_Search for latest drug safety alerts if no relevant doc found in RAG_Search\n"
                "3. Check for interaction warnings\n\n"
                "**CRITICAL SAFETY:**\n"
                "- NEVER provide specific dosing without prescription context\n"
                "- Always warn about pregnancy/breastfeeding considerations\n"
                "- Flag potential allergic reactions\n"
                "- Recommend pharmacist/doctor consultation for dosing\n\n"
                "**OUTPUT FORMAT:**\n"
                "1. Medication Overview\n"
                "2. Common Uses\n"
                "3. Important Warnings & Interactions\n"
                "4. When to Consult Healthcare Provider\n"
            ),
            model_name="gemini/gemini-2.5-flash",
            max_loops=3,
            react_on=False,
            verbose=False,
            tools=[medical_rag_tool, medical_web, self.general_web],
            temperature=0.3,
            dynamic_temperature_enabled=False,
            stopping_func=stop_when_no_answer_generated,
            tool_retry_attempts=1,
            tool_call_summary=True
        )
        
        # ðŸ¥— 4. NUTRITION AGENT - Diet and lifestyle
        nutrition_agent = Agent(
            agent_name="Nutrition_Agent",
            agent_description="Registered Dietitian specializing in nutrition and lifestyle",
            system_prompt=(
                "You are a Registered Dietitian AI assistant.\n\n"
                "**EXPERTISE:**\n"
                "- Nutritional deficiencies and dietary solutions\n"
                "- Disease-specific diets (diabetes, hypertension, etc.)\n"
                "- Vitamin/mineral supplementation\n"
                "- Healthy eating patterns and meal planning\n\n"
                "**WORKFLOW:**\n"
                "1. Assess nutritional context of query\n"
                "2. RAG_Search for dietary guidelines\n"
                "3. Use Medical_Web_Search if no relevant doc found in RAG_Search and for evidence-based nutrition\n\n"
                "**APPROACH:**\n"
                "- Focus on whole foods over supplements when possible\n"
                "- Consider cultural and dietary restrictions\n"
                "- Provide practical, sustainable advice\n"
                "- Flag when medical nutrition therapy (MNT) is needed\n\n"
                "**OUTPUT FORMAT:**\n"
                "1. Nutritional Assessment\n"
                "2. Dietary Recommendations\n"
                "3. Foods to Emphasize/Limit\n"
                "4. When to See a Dietitian\n"
            ),
            model_name="gemini/gemini-2.5-flash",
            max_loops=3,
            react_on=False,
            verbose=False,
            tools=[medical_rag_tool, medical_web, self.general_web],
            temperature=0.3,
            dynamic_temperature_enabled=False,
            stopping_func=stop_when_no_answer_generated,
            tool_retry_attempts=1,
            tool_call_summary=True
        )
        
        
        self.expert_agents.update({
            'gp': gp_agent,
            'pharmacist': pharmacist_agent,
            'nutrition': nutrition_agent
        })
        
        logger.info("âœ… Enhanced Medical experts created")
        return self.expert_agents
    
    def create_insurance_experts(self) -> Dict[str, Agent]:
        """Insurance domain - RAG only for proprietary docs"""
        
        # Etiqa Agent (HAS RAG - proprietary policies)
        etiqa_rag = SpecializedRAGPipeline(
            collection_name="etiqa_policies",
            persist_directory=self.persist_directory,
            domain="insurance",
            specialty="etiqa"
        )
        self.rag_pipelines['etiqa'] = etiqa_rag
        etiqa_rag_tool = self.tool_factory.create_rag_tool(etiqa_rag, self.parent_qa_system)
        
        etiqa_agent = Agent(
            agent_name="Etiqa_Agent",
            system_prompt=(
                "You are an Etiqa Takaful specialist.\n"
                "1. RAG_Search for policy documents\n"
                "2. Etiqa_Web_Search for website info\n"
                "3. Cite policy clauses explicitly\n"
            ),
            model_name="gemini/gemini-2.5-flash",
            max_loops=2,
            verbose=False,
            tools=[etiqa_rag_tool, self.etiqa_web, self.general_web],
            temperature=0.1,
            stopping_func=stop_when_no_answer_generated,
            tool_retry_attempts=1,
            tool_call_summary=True
        )
        
        # General Insurance Agent (NO RAG - general knowledge)
        general_insurance_agent = Agent(
            agent_name="General_Insurance_Agent",
            system_prompt=(
                "You are a general insurance expert aside from Etiqa company.\n"
                "Use your knowledge of insurance concepts + General_Web_Search.\n"
                "Explain takaful, premiums, coverage types clearly.\n"
            ),
            model_name="gemini/gemini-2.5-flash",
            max_loops=2,
            verbose=False,
            tools=[self.general_web],
            temperature=0.1,
            stopping_func=stop_when_no_answer_generated,
            tool_retry_attempts=1,
            tool_call_summary=True
        )
        
        self.expert_agents.update({
            "etiqa": etiqa_agent,
            "general_insurance": general_insurance_agent
        })
        
        logger.info("âœ… Insurance experts created")
        return self.expert_agents
    
    def create_summarizer_agent(self) -> Agent:
        summarizer = Agent(
            agent_name="Summarizer_Agent",
            system_prompt = (
            "You are a expert synthesis assistant. Your task is to merge multiple expert responses into one clear, actionable, and trustworthy summary for a general citizen.\n\n"

            "**FIRST STEP - LENGTH MANAGEMENT:**\n"
            "- The manager will specify a target length in the 'word_limit' field. STRICTLY adhere to it.\n"
            "- If 'word_limit' is absent, keep the final output under 250 words.\n"
            "- **Count your words.** Your final output must never exceed the limit.\n\n"

            "**CORE PRINCIPLES for All Domains (Medical, Islamic, Insurance, etc.):**\n"
            "1.  **Clarity & Simplicity:** Use everyday language. Explain necessary technical terms (e.g., 'premium,' 'hadith,' 'hypertension') in plain words within parentheses.\n"
            "2.  **Empathy & Reassurance:** Acknowledge the user's situation. Use a tone that is helpful, not alarming.\n"
            "3.  **Action-Oriented:** Focus on what the citizen can or should do. Prioritize practical steps.\n"
            "4.  **Neutral Synthesis:** If experts disagree, present the consensus or the range of reputable opinions clearly without taking sides. Do not amplify conflict.\n"
            "5.  **Source Transparency:** You are an aggregator, not an original source. Always attribute information.\n\n"

            # "**REQUIRED OUTPUT STRUCTURE:**\n"
            # "Follow this format exactly. Do not add extra sections or commentary.\n\n"

            # "**Direct Answer**\n"
            # "Begin with a 2-3 sentence, high-level summary that directly answers the citizen's core concern. Be reassuring and highlight the most critical action.\n\n"

            # "**Detailed Explanation**\n"
            # "Synthesize the key points from all experts into a flowing narrative.\n"
            # "- **Integrate Citations Smoothly:** Weave source references into the text (e.g., 'According to the NHS...', 'As noted in *Umdat al-Salik*...').\n"
            # "- **Simplify Expert Language:** Rephrase complex points for clarity, but never change the original meaning or risk level.\n"
            # "- **Organize Logically:** Group related information. Use paragraphs for readability.\n\n"

            # "**Key Takeaways**\n"
            # "Provide exactly 3 bullet points. Each must be:\n"
            # "- **Actionable** (What the citizen should do)\n"
            # "- **Memorable** (Clear and concise)\n"
            # "- **Self-contained** (Understandable on its own)\n\n"

            # "**References**\n"
            # "List every source cited in the 'Detailed Explanation' in a consistent, simple format.\n"
            # "- **For Islamic Sources:** Cite the actual book/compilation (e.g., *Sahih al-Bukhari*, *Minhaj al-Talibin*).\n"
            # "- **For Medical/Insurance:** Cite the guideline, institution, or article (e.g., NHS, Mayo Clinic, Policy Document XYZ).\n"
            # "- **Dont cite the agent name such as (Pharmacist_Agent & Maliki_Agent and etc), cite the sources that the agents used to answer the question (e.g., Umdat al-Salik,NHS,Mayoclinic)."
            # "- **CRITICAL RULE:** Only list sources provided by the experts. If a source is missing, omit it; never invent or guess a source name.\n\n"

            "**FINAL CHECKS BEFORE OUTPUT:**\n"
            "- ✅ Word limit obeyed.\n"
            "- ✅ Tone is empathetic and citizen-friendly.\n"
            "- ✅ All critical expert advice is included.\n"
            "- ✅ Sources are correctly attributed and listed.\n"
            "- ✅ The structure is followed exactly.\n"
            "- ✅ The output is a coherent whole, not a disjointed list.\n"
        ),
            model_name="gemini/gemini-2.5-flash",
            max_loops=1,
            verbose=True,
            react_on=False,
            tools=None,
            stopping_func=None
        )
        
        self.expert_agents['summarizer'] = summarizer
        return summarizer

# ============================================================================
# MAIN QA SYSTEM 
# ============================================================================

class MixtureOfAgentsQA:
    def __init__(self, domain: str, persist_directory: str = "./chroma_db_swarm"):
        self.domain = domain
        self.persist_directory = persist_directory
        self.current_k = 5
        self.conversation_history = []  
        self.eval_logs = []  
        self.eval_csv_file = f"./rag_eval_{domain}_{datetime.now().strftime('%Y%m%d')}.csv"
        self._init_eval_csv()
        
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1
        )
        
        self.expert_factory = OptimizedExpertFactory(self.llm, persist_directory)
        self.expert_factory.parent_qa_system = self  # Link back for dynamic k
        self.vision_processor = SimpleVisionProcessor(self.llm, domain)
        
        self._create_experts_for_domain()
        self._create_orchestration_agents()
        self.query_refiner = Agent(
            agent_name="Query_Refiner",
            system_prompt=(
                "You refine follow-up queries using conversation history.\n\n"
                "If history does not relate to the query,return the query as it is or just rephrase the query"
                "**TASK:** Rewrite ambiguous follow-ups into standalone queries.\n\n"
                "**EXAMPLES:**\n"
                "User(history): Q: 'What is wudu?' Assistant(history): 'Ritual washing...'\n"
                "Follow-up: 'What breaks it?' ->' Refined: 'What breaks wudu in Islamic fiqh?'\n\n"
                "User(history): Q: 'Headaches for 2 weeks' Assistant(history): 'Could be tension...'\n"
                "Follow-up: 'What about medication?' ->' Refined: 'What medications help tension headaches?'\n\n"
                "**OUTPUT:** Only the refined query, nothing else.\n"
            ),
            model_name="gemini/gemini-2.5-flash",
            max_loops=1,
            react_on=False,
            verbose=False,
            tools=None
        )
        
        logger.info(f"âœ… Optimized QA System initialized for {domain}")
        
    def _clean_moa_output(self, raw_output: str) -> str:
        """
        Clean MoA output by removing system metadata and extracting only the final answer.
        
        Args:
            raw_output: Raw string from MoA.run()
            
        Returns:
            Cleaned final answer
        """
        
        # Pattern 1: Remove "System: Team Name..." metadata
        # This appears when MoA includes its system info
        pattern1 = r"System:\s*Team Name:.*?(?=\*\*Direct Answer\*\*|\*\*Answer\*\*|$)"
        cleaned = re.sub(pattern1, "", raw_output, flags=re.DOTALL | re.IGNORECASE)
        
        # Pattern 2: If multiple "Direct Answer" sections exist, take only the LAST one
        # (This handles accumulated responses from previous queries)
        direct_answer_pattern = r"\*\*\s*Direct Answer\s*\*\*"
        matches = list(re.finditer(direct_answer_pattern, cleaned, re.IGNORECASE))
        
        if len(matches) > 1:
            # Multiple "Direct Answer" sections found - take from the last one
            last_match = matches[-1]
            cleaned = cleaned[last_match.start():]
            logger.warning(f"⚠️ Found {len(matches)} answer sections, using only the last one")
        
        # Pattern 3: Remove any leading/trailing whitespace
        cleaned = cleaned.strip()
        
        # Pattern 4: If still too long (> 5000 chars), something's wrong
        if len(cleaned) > 5000:
            logger.warning(f"⚠️ Answer unusually long ({len(cleaned)} chars), might contain accumulated responses")
            
            # Emergency extraction: Find the last complete answer structure
            # Look for the pattern: **Direct Answer** ... **References**
            last_answer_pattern = r"(\*\*\s*Direct Answer\s*\*\*.*?)(?=\*\*\s*Direct Answer\s*\*\*|$)"
            emergency_matches = list(re.finditer(last_answer_pattern, cleaned, re.DOTALL | re.IGNORECASE))
            
            if emergency_matches:
                cleaned = emergency_matches[-1].group(1).strip()
                logger.info(f"✅ Emergency extraction successful, reduced to {len(cleaned)} chars")
        
        return cleaned
    
    def _init_eval_csv(self):
        """Initialize CSV file with headers"""
        os.makedirs(os.path.dirname(self.eval_csv_file), exist_ok=True)
        file_exists = os.path.isfile(self.eval_csv_file)
        
        with open(self.eval_csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Write headers
                writer.writerow([
                    'timestamp', 'query', 'doc_count', 
                    'context', 'metadata', 'specialty'
                ])
    
    def _save_eval_logs_to_csv(self, query: str):
        """Save each RAG call as a row in CSV"""
        if not self.eval_logs:
            return
        
        with open(self.eval_csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            for rag_call in self.eval_logs:
                for doc in rag_call.get('docs', []):
                    writer.writerow([
                        rag_call.get('timestamp', datetime.now().isoformat()),
                        rag_call.get('query', query),
                        rag_call.get('doc_count', 0),
                        doc.get('content', ''),  # Preview
                        json.dumps(doc.get('metadata', {})),
                        doc.get('metadata', {}).get('specialty', 'unknown')
                    ])
        
        logger.info(f"ðŸ“Š Appended {len(self.eval_logs)} RAG calls to CSV")
    def _refine_query_with_history(self, query: str, chat_history: list) -> str:
        """Refine query using conversation context"""
        if not chat_history or len(chat_history) == 0:
            return query
        
        # Format last 3 exchanges (6 messages max) to avoid token bloat
        recent_history = chat_history[-4:]
        history_text = "\n".join([
            f"{'User' if i % 2 == 0 else 'Assistant'}: {msg['content'][:200]}..."  # Truncate long messages
            for i, msg in enumerate(recent_history)
        ])
        
        refiner_prompt = f"""
        **CONVERSATION HISTORY:**
        {history_text}

        **NEW QUERY:** {query}

        **TASK:** If the new query is a follow-up (uses "it", "this", "that", "what about"), rewrite it as a standalone question. Otherwise, return it as-is.

        **REFINED QUERY:**
        """
                
        refined = self.query_refiner.run(refiner_prompt).strip()
        # Take only the last line (the actual refined query)

        if refined:
            # Clean multi-line LLM output
            clean = [line.strip() for line in refined.split("\n") if line.strip()][-1]

            logger.info(f"ðŸ”„ Query refined: '{query}' vs '{clean}'")
            return clean
        
        return query
        
    
    def _create_experts_for_domain(self):
        if self.domain == "islamic":
            self.experts = self.expert_factory.create_islamic_experts()
            self.expert_list = ['shafii', 'maliki', 'hanafi_hanbali', 'hadith', 'general_islamic']
        elif self.domain == "medical":
            self.experts = self.expert_factory.create_medical_experts()
            self.expert_list = ['gp', 'pharmacist', 'nutrition']
        elif self.domain == "insurance":
            self.experts = self.expert_factory.create_insurance_experts()
            self.expert_list = ['etiqa', 'general_insurance']
    
    def _create_orchestration_agents(self):
        """Fixed version - no more memory bleeding!"""
        
        # NEW: Use LangChain manager
        self.manager_agent = SimpleLangChainManager(
            domain=self.domain,
            expert_list=self.expert_list
        )
        
        # Keep existing summarizer
        self.summarizer = self.expert_factory.create_summarizer_agent()
        
    
    def answer(self, query: str, chat_history: list = None, image_path: str = None) -> dict:
        start_time = time.time()  
        self.eval_logs = []         
        if chat_history is None:
            chat_history = []
        
        # Refine query with history (keep existing code)
        refined_query = query
        if chat_history:
            recent_history = chat_history[-4:]
            refined_query = self._refine_query_with_history(query, recent_history)
        print(f"Refined query {refined_query}")
        logger.info(f"\nðŸš€ Processing: '{refined_query}'")
        
        # Image processing (keep existing code)
        query_with_context = refined_query
        image_description = None
        
        if image_path:
            image_description = self.vision_processor.process_image(image_path)
            query_with_context = f"{refined_query}\n\n[IMAGE CONTEXT]: {image_description}"
        
        # âœ… FIXED: Use new LangChain manager (replaces old manager code)
        analysis = self.manager_agent.route(query_with_context)
        
        experts_list = analysis['experts']
        complexity = analysis['complexity']
        
        # Set word limit (keep existing code)
        word_limits = {"simple": 110, "moderate": 250, "complex": 350}
        word_limit = word_limits.get(complexity, 200)
        
        print(f"Complexity: {complexity}, Experts: {experts_list}")
        
        # IMPORTANT: Clear agent memory before running
        active_agents = [self.experts[name] for name in experts_list]
        
        # for agent in active_agents:
        #     if hasattr(agent, 'short_memory'):
        #         agent.short_memory.clear()
        
        # # Clear summarizer memory too
        # if hasattr(self.summarizer, 'short_memory'):
        #     self.summarizer.short_memory.clear()
        
        # ================================================
        # ðŸš€ PARALLEL EXECUTION (Instead of Sequential MoA)
        # ================================================
        # if len(active_agents) > 1:
        #     logger.info("ðŸ”¥ Running agents in parallel...")
            
        #     # Create parallel executor
        #     parallel_executor = ParallelSwarmExecutor(timeout_per_agent=45)
            
        #     # Run all agents in parallel
        #     agent_responses = parallel_executor.execute_parallel(
        #         agents=active_agents, 
        #         query=query_with_context
        #     )
            
        #     # Prepare context for summarizer
        #     response_context = "\n\n".join([
        #         f"## {agent.agent_name}:\n{response}"
        #         for agent, response in zip(active_agents, agent_responses.values())
        #     ])
            
        #     # Create summarizer prompt
        #     summarizer_prompt = f"""
        #     **Original Query:** {refined_query}
        #     **Word Limit:** {word_limit} words
            
        #     **Expert Responses:**
        #     {response_context}
            
        #     Please synthesize the above expert responses into a single, coherent answer.
        #     Follow the citizen-friendly format:
        #     1. Direct Answer (brief)
        #     2. Detailed Explanation (simplify technical terms)
        #     3. Key Takeaways (2-3 bullet points)
        #     4. References (cite all sources used)
            
        #     **IMPORTANT:** Do not exceed {word_limit} words total.
        #     """
            
        #     # Get final answer from summarizer
        #     final_answer = self.summarizer.run(summarizer_prompt)
        #     source = f"Parallel ({', '.join(experts_list)})"
            
        # else:
        #     # Single agent case (no parallel needed)
        #     logger.info(f"ðŸ‘¤ Single agent: {experts_list[0]}")
        #     response = active_agents[0].run(query_with_context)
            
        #     summarizer_prompt = f"""
        #     **Query:** {refined_query}
        #     **Expert ({experts_list[0]}) Response:** {response}

        #     Format into citizen-friendly answer (max {word_limit} words):
        #     1. Direct Answer
        #     2. Explanation (preserve citations)
        #     3. Key Takeaways (2-3 bullets)
        #     4. References
        #     """
        #     final_answer = self.summarizer.run(summarizer_prompt)
        #     source = f"Single ({experts_list[0]})"
        if len(active_agents) > 1:
            logger.info("ðŸ”¥ Running MoA...")
            
            moa = MixtureOfAgents(
                agents=active_agents,
                aggregator_agent=self.summarizer,
                layers=1
            )
            moa_query = f"[WORD_LIMIT={word_limit}] {query_with_context}"
            
            moa_output = moa.run(moa_query)

            # STEP 3: Comprehensive extraction
            print("\n" + "="*70)
            print("🔍 MoA OUTPUT TYPE:", type(moa_output))
            print("="*70)
            print(f"📏 MoA length: {len(moa_output)}")
            # Try multiple extraction methods
            final_answer = None

            if isinstance(moa_output, str):
                final_answer = moa_output
                print("✅ Extracted as string")

            final_answer = self._clean_moa_output(moa_output)
            
            print(f"📏 Final answer length: {len(final_answer)}")
            print(f"📄 Preview: {final_answer[:200]}...")
            print("="*70 + "\n")

            # STEP 4: Now clear memory AFTER extraction
            for agent in active_agents:
                if hasattr(agent, 'short_memory'):
                    agent.short_memory.clear()

            if hasattr(self.summarizer, 'short_memory'):
                self.summarizer.short_memory.clear()
            source = f"MoA ({', '.join(experts_list)})"
        else:
            logger.info(f"ðŸ‘¤ Single agent: {experts_list[0]}")
            response = active_agents[0].run(query_with_context)
            summarizer_prompt = f"""
            **Query:** {refined_query}
            **Expert ({experts_list[0]}) Response:** {response}

            Format into citizen-friendly answer (max {word_limit} words):
            1. Direct Answer
            2. Explanation (preserve citations)
            3. Key Takeaways (2-3 bullets)
            4. References
            """
            final_answer = self.summarizer.run(summarizer_prompt)
            source = f"Single ({experts_list[0]})"
            
        # clean = extract_from_bold_answer(final_answer)
        # print(f"Clean {clean}")
        # final_answer = markdown_to_html(final_answer)
        print(f"Final answer :{final_answer}")
        # Update chat history
        total_time = time.time() - start_time
        print(f"â±ï¸ Total execution time: {total_time:.2f} seconds")
        updated_history = chat_history.copy()
        updated_history.append({"role": "user", "content": refined_query})
        updated_history.append({"role": "assistant", "content": final_answer})
        
        self._save_eval_logs_to_csv(refined_query)
        return {
            "answer": final_answer,
            "experts_consulted": experts_list,
            "complexity": complexity,
            "reasoning": analysis["reasoning"],
            "source": source,
            "refined_query": refined_query,
            "chat_history": updated_history,
            "eval_logs": self.eval_logs
        }
        
class SimpleVisionProcessor:
    """Lightweight domain-specific image analysis"""
    
    def __init__(self, llm, domain: str):
        self.llm = llm
        self.domain = domain
        
        # Simple vision prompts (ONE SENTENCE like your original)
        self.vision_prompts = {
            "medical": "Describe this medical image in one sentence, focusing on visible symptoms or affected area.",
            "islamic": "Describe this Islamic image in one sentence, including any Arabic text or religious context.",
        }
    
    def process_image(self, image_path: str) -> str:
        """
        Process image and return description string.
        
        Args:
            image_path: Path to image file
            
        Returns:
            str: Image description or error message
        """
        try:
            # Encode image to base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Get domain-specific prompt
            vision_prompt = self.vision_prompts.get(
                self.domain, 
                "Describe this image in one sentence."
            )
            
            # Create vision message
            vision_message = HumanMessage(
                content=[
                    {"type": "text", "text": vision_prompt},
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
                ]
            )
            
            # Get vision response
            vision_response = self.llm.invoke([vision_message])
            description = vision_response.content.strip()
            
            logger.info(f"ðŸ–¼ï¸ Vision processed: {description[:100]}...")
            return description
            
        except FileNotFoundError:
            error_msg = f"Image file not found: {image_path}"
            logger.error(f"âŒ {error_msg}")
            return f"[Error: {error_msg}]"
        except Exception as e:
            error_msg = f"Vision processing failed: {str(e)}"
            logger.error(f"âŒ {error_msg}")
            return f"[Error: {error_msg}]"
        
# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    PERSIST_DB = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "chroma_db_swarm"))
    
# # #     # Islamic test
    islamic_qa = MixtureOfAgentsQA(domain="islamic", persist_directory=PERSIST_DB)
    # insurance_qa = MixtureOfAgentsQA(domain="insurance", persist_directory=PERSIST_DB)
    # medical_qa = MixtureOfAgentsQA(domain="medical",persist_directory=PERSIST_DB)
#     query=["What is the ruling on touching the Quran without wudu ?","Is it necessary to recite fatihah behind imam in maliki school"]
    result = islamic_qa.answer(
            query="Salam, I got a job offer as a 3D artist at a gaming company in Cyberjaya. My job is to design characters and environments for mobile games. Some games have in-app purchases (loot boxes) and some have fantasy elements like magic. Is my salary halal? What if I only work on the art side and not the gambling mechanics? Can I take this job or should I reject it?"
        )
        
    # result = insurance_qa.answer(
    #     query="What is the Wakalah fee and what does it include?"
    # )
    # result = medical_qa.answer(  
    #     query = "Can eating yogurt everyday helps reduce my blood pressure since i have high blood pressure. "
    # )

#     print(res)
    # print(f"\nðŸ‘¥ Experts: {', '.join(result['experts_consulted'])}")
    
     # Conversation 1
    # history = []
    # result = medical_qa.answer(
    #     query="My mum is 68 years old and just started taking statins (Atorvastatin 20mg) for her cholesterol. But now she's complaining her muscles feel weak and painful, especially her legs. Is this normal? Should she continue taking it? Also, what kind of food can help lower cholesterol naturally? She loves eating nasi kandar and roti canai every morning - does she need to stop completely?"
    # )
    # # print(result1["answer"])
    # history = result1["chat_history"]  # Update history
    
    # Conversation 2 (follow-up)
    # result2 = medical_qa.answer(
    #     query="What about medication for this?",  # Ambiguous
    #     chat_history=history
    # )
    
    # result2 = islamic_qa.answer(
    #     query="What could this place be?",
    #     image_path="gambo.jpg"
    # )
    # print("\nImage Analysis:", result2["image_description"])
    # print("Medical Response:", result2[
    print("\n📊 FINAL SYNTHESIZED ANSWER:")
    print(result['answer'])