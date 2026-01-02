from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.tools import Tool
from langchain_tavily import TavilySearch
from langchain_community.retrievers import BM25Retriever
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_core.agents import AgentAction
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from flashrank import Ranker, RerankRequest
from src.metrics_tracker import MetricsTracker
import logging


# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# class ContextRetriever:
#     def __init__(self, retriever):
#         self.retriever = retriever

#     def deduplicate_context(self, context_list): 
#         """Deduplicate context entries to avoid repetition."""
#         seen = set()
#         deduped = []
#         for item in context_list:
#             if item not in seen:
#                 seen.add(item)
#                 deduped.append(item)
#         return "\n".join(deduped) if deduped else "No relevant context found."

#     def retrieve(self, query, top_k=5):
#         """
#         Retrieve the top-k relevant contexts from ChromaDB based on the query.
        
#         Args:
#             query (str): The query or prediction to search for.
#             top_k (int): Number of top results to return (default: 3).
        
#         Returns:
#             str: Deduplicated context string from the top-k results.
#         """
#         logger.info(f"Retrieving for query: {query}")
#         try:
#             # Perform similarity search using ChromaDB retriever
#             results = self.retriever.invoke(query, k=top_k) 
#             logger.info(f"Retrieved documents: {[doc.metadata.get('source') for doc in results]}")
            
#             # Extract the page content (context) from each result
#             contexts = [doc.page_content for doc in results]
#             logger.info(f"Context : {contexts}")
            
#             # Deduplicate the contexts
#             deduped_context = self.deduplicate_context(contexts)
#             logger.info(f"Deduplicated context: {deduped_context}")
            
#             return deduped_context
#         except Exception as e:
#             logger.error(f"Retrieval error: {str(e)}")
#             return "Retrieval failed due to error."

class LLMComplexityAnalyzer:
    """
    Analyzes query complexity using an LLM to make a "managerial" decision
    on the optimal retrieval strategy.
    """
    
    def __init__(self, domain: str, llm: ChatGoogleGenerativeAI):
        self.domain = domain
        self.llm = llm
        
        self.system_prompt = (
            "You are a 'Complexity Analyzer' manager for a RAG (Retrieval-Augmented Generation) system. "
            "Your domain of expertise is: **{domain}**."
            "\n"
            "Your task is to analyze the user's query and determine its complexity. Based on this, "
            "you will decide how many documents (k) to retrieve. More complex queries require "
            "more documents to synthesize a good answer."
            "\n"
            "Here are the retrieval strategies:"
            "1.  **simple**: For simple, direct fact-finding queries. (e.g., 'What is takaful?') "
            "    - Set k = 5"
            "2.  **moderate**: For queries that require explanation, some comparison, or have multiple parts. "
            "    (e.g., 'What is the difference between madhab Shafi'i and Maliki on prayer?') "
            "    - Set k = 7"
            "3.  **complex**: For deep, nuanced, multi-step, or highly comparative/synthetic queries. "
            "    (e.g., 'Explain in detail the treatment options for type 2 diabetes, comparing "
            "    their side effects and suitability for elderly patients.')"
            "    - Set k = 10"
            "\n"
            "Analyze the following query and provide your reasoning."
            "\n"
            "**IMPORTANT**: You MUST respond ONLY with a single, valid JSON object. Do not add any "
            "other text. The JSON object must have these three keys:"
            "-   `complexity`: (string) Must be one of 'simple', 'moderate', or 'complex'."
            "-   `k`: (integer) Must be 5, 10, or 15, corresponding to the complexity."
            "-   `reasoning`: (string) A brief 1-sentence explanation for your decision."
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt.format(domain=self.domain)),
            ("human", "{query}")
        ])
        
        self.output_parser = JsonOutputParser()
        
        # This chain will output a parsed dictionary
        self.chain = self.prompt_template | self.llm | self.output_parser
        
        logger.info(f"🧠 LLMComplexityAnalyzer initialized for '{self.domain}'")

    def analyze(self, query: str) -> dict:
        """
        Analyzes query complexity using an LLM and returns the retrieval strategy.
        """
        logger.info(f"🧠 LLMComplexityAnalyzer: Analyzing query...")
        
        try:
            # Invoke the chain to get the structured JSON output
            result = self.chain.invoke({"query": query})
            
            # Add a 'score' field for compatibility
            score_map = {"simple": 2, "moderate": 4, "complex": 6}
            result['score'] = score_map.get(result.get('complexity'), 0)
                
            logger.info(f"🧠 LLM Decision: {result.get('complexity').upper()} (k={result.get('k')})")
            logger.info(f"   Reasoning: {result.get('reasoning')}")
            
            return result
            
        except Exception as e:
            # Fallback in case the LLM fails or returns bad JSON
            logger.error(f"❌ LLMComplexityAnalyzer failed: {e}. Defaulting to 'moderate' strategy.")
            return {
                "complexity": "moderate",
                "k": 12,
                "score": 4,
                "reasoning": "Fallback: LLM analysis or JSON parsing failed."
            }


class HybridRetriever:
    """
    Multi-retriever swarm that executes parallel retrieval strategies.
    Worker component that takes orders from LLMComplexityAnalyzer.
    """
    
    def __init__(self, chroma_retriever, documents):
        self.dense_retriever = chroma_retriever  # Semantic search
        self.bm25_retriever = BM25Retriever.from_documents(documents)  # Keyword search
        self.bm25_retriever.k = 20  # Set high, will be limited by k parameter
        logger.info("✅ Hybrid Retriever initialized (Dense + BM25 workers)")
    
    def retrieve_with_swarm(self, query: str, k: int) -> list:
        """
        Execute multi-retriever swarm with parallel workers.
        """
        logger.info(f"🐝 Swarm deployment: {2} workers, target k={k}")
        
        # Define worker tasks
        retrieval_tasks = {
            "dense_semantic": lambda: self.dense_retriever.invoke(query, k=k),
            "bm25_keyword": lambda: self.bm25_retriever.invoke(query)[:k],
        }
        
        # Execute workers in parallel
        swarm_results = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(task): name 
                for name, task in retrieval_tasks.items()
            }
            
            for future in as_completed(futures):
                worker_name = futures[future]
                try:
                    results = future.result()
                    swarm_results[worker_name] = results
                    logger.info(f"  ✅ Worker '{worker_name}': {len(results)} docs")
                except Exception as e:
                    logger.error(f"  ❌ Worker '{worker_name}' failed: {e}")
                    swarm_results[worker_name] = []
        
        # Combine and deduplicate documents
        combined_docs = self._combine_and_deduplicate(swarm_results)
        
        return combined_docs
    
    def _combine_and_deduplicate(self, swarm_results: dict) -> list:
        """Combine results from all workers and remove duplicates."""
        all_docs = []
        seen_content = set()
        worker_contributions = {}
        
        for worker_name, docs in swarm_results.items():
            for doc in docs:
                # Use first 200 chars as hash to detect duplicates
                content_hash = hash(doc.page_content[:200])
                
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    
                    # Tag document with source worker
                    doc.metadata['swarm_worker'] = worker_name
                    all_docs.append(doc)
                    
                    # Track contributions
                    worker_contributions[worker_name] = \
                        worker_contributions.get(worker_name, 0) + 1
        
        logger.info(f"🐝 Swarm combined: {len(all_docs)} unique docs")
        logger.info(f"   Worker contributions: {worker_contributions}")
        
        return all_docs
    
class AgenticQA:
    def __init__(self, config=None):  
        logger.info("Initializing AgenticQA...")
        
        # Load a small, fast reranker model. This runs locally.
        try:
            self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
            logger.info("FlashRank Reranker loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load FlashRank reranker: {e}")
            self.reranker = None
        
        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. "
            "IMPORTANT: DO NOT provide any answers or explanations. ONLY rephrase the question if needed. "
            "If the question is already clear and standalone, return it exactly as is. "
            "Output ONLY the reformulated question, nothing else."
        )
        
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [("system", self.contextualize_q_system_prompt),
             MessagesPlaceholder("chat_history"),
             ("human", "{input}")]
        )
        self.qa_system_prompt = (
            "You are an assistant that answers questions in a specific domain for citizens mainly in Malaysia, "
            "depending on the context. "
            "You will receive:\n"
            "  • domain = {domain}  (either 'medical', 'islamic' , or 'insurance')\n"
            "  • context = relevant retrieved passages\n"
            "  • user question\n\n"
            "If the context does not contain the answer, **YOU MUST SAY 'I do not know'** or 'I cannot find that information in the provided documents.' Do not use your general knowledge.\n\n"
            "Instructions based on domain:\n"
            "1. If domain = 'medical' :\n"
            "   - Answer the question in clear, simple layperson language, "
            "   - Citing your sources (e.g. article name, section)."
            "   - Add a medical disclaimer: “I am not a doctor…”.\n"
            "2. If domain = 'islamic':\n"
            " Always put citations or sources at the end of final answer"
            "   - **ALWAYS present both Shafi'i AND Maliki perspectives** if the question is about fiqh/rulings\n"
            "   - **Cite specific sources**: Always mention the book name (e.g., 'According to Muwatta Imam Malik...', 'Minhaj al-Talibin states...', 'Umdat al-Salik explains...')\n"
            "   - **Structure answer as**:\n" 
            "      - Shafi'i view (from Umdat al-Salik/Minhaj): [ruling with citation]\n"
            "      - Maliki view (from Muwatta): [ruling with citation]\n"
            "      - If they agree: mention the consensus\n"
            "      - If they differ: present both views objectively without favoring one\n"
            "   - **For hadith questions**: provide the narration text, source (book name, hadith number)\n "
            "   - - **If ruling has EXCEPTIONS** (like 'except for...', 'unless...'), YOU MUST include them. "
            "      If context doesn't show exceptions but the ruling seems absolute, indicate this uncertainty.\n"
            "   - If the context does not contain relevant information from BOTH madhabs, acknowledge which sources you have "
            "      (e.g., 'Based on Shafi'i sources only...') and suggest consulting additional madhab resources.\n"
            "   - **Always end with**: 'This is not a fatwa. Consult a local scholar for guidance specific to your situation.'\n"
            "   - Always include hadith narration or quran verse as evidence (if it exists) in the final response "
            "   - Keep answers concise but comprehensive enough to show different scholarly views.\n\n"
            
            "3. If domain = 'insurance':\n"
            "   - Your knowledge is STRICTLY limited to Etiqa Takaful (Motor and Car policies).\n"
            "   - First, try to answer ONLY using the provided <context>.\n"
            "   - **If the answer is not in the context, YOU MUST SAY 'I do not have information on that specific topic.'** Do not make up an answer.\n"
            "   - If the user asks about other Etiqa products (e.g., medical, travel), you MUST use the 'EtiqaWebSearch' tool.\n"
            "   - If the user asks about another insurance company (e.g., 'Prudential', 'Takaful Ikhlas'), state that you can only answer about Etiqa Takaful.\n"
            "   - If the user asks a general insurance question (e.g., 'What is takaful?', 'What is an excess?'), use the 'GeneralWebSearch' tool.\n"
            
            "4. IMPORTANT: Follow this structure for FINAL ANSWER: for all domains:\n"
            "   Direct Answer (short, plain-language summary)\n"
            "   Detailed Explanation (preserve citations, but simplify wording)\n"
            "   Key Takeaways (2-3 bullets, actionable and easy to scan)\n"
            "   References: Provide a clear list of all sources.Cite actual books names like Umdat-al Salik ,Minhaj At Talibin and so on, articles, or websites.\n\n" 
            "5. For ALL domains: If the context does not contain the answer, do not make one up. Be honest.\n\n"
            "Context:\n"
            "{context}"
            )

        self.qa_prompt = ChatPromptTemplate.from_messages(
            [("system", self.qa_system_prompt),
             MessagesPlaceholder("chat_history"),
             ("human", "{input}")]
        )
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.05) 
        # --- START: NEW QUERY REFINER ---
        self.refiner_system_prompt = (
            "You are an expert search query refiner. Your task is to take a user's question "
            "and rewrite it to be a perfect, concise search query for a database. "
            "Remove all conversational fluff, emotion, and filler words. "
            "Distill the query to its core semantic intent. "
            "For example:"
            "- 'Hi, I was wondering if I can touch a dog if I found it is cute?' becomes 'ruling on touching a dog in islam'"
            "- 'What are the treatments for, like, a common cold?' becomes 'common cold treatment options'"
            "- 'Tell me about diabetes' becomes 'what is diabetes'"
            "Output ONLY the refined query, nothing else."
        )
        
        self.refiner_prompt = ChatPromptTemplate.from_messages([
            ("system", self.refiner_system_prompt),
            ("human", "{query}")
        ])
        
        self.refiner_chain = self.refiner_prompt | self.llm
        logger.info("✅ Query Refiner chain initialized.")
        # --- END: NEW QUERY REFINER ---

        self.react_docstore_prompt = hub.pull("aallali/react_tool_priority")
        self.answer_validator = AnswerValidatorAgent(self.llm)

        self.retriever = None
        self.agent_executor = None
        self.tools = [] # Initialize the attribute
        self.domain = "general"
        self.answer_validator = None
        self.retrieval_agent = None

        if config:
            logger.info(f"Configuring AgenticQA with provided config: {config}")
            try:
                collection_name = config["retriever"]["collection_name"]
                persist_directory = config["retriever"]["persist_directory"]
                self.domain = config.get("domain", "general") # Get domain from config
                
                # 1. Initialize the embedding function
                embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

                # 2. Connect to the persistent ChromaDB
                db_client = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embedding_function,
                    collection_name=collection_name
                )

                # 3. Set the retriever for this instance
                self.retriever = db_client.as_retriever()
                logger.info(f"✅ Successfully created retriever for collection '{collection_name}'")
                # --- START: NEW SWARM INITIALIZATION ---
                logger.info("Initializing Swarm components...")
                # Get all documents from Chroma for BM25
                all_docs_data = db_client.get()
                docs_for_bm25 = [
                    Document(page_content=content, metadata=meta)
                    for content, meta in zip(
                        all_docs_data['documents'], 
                        all_docs_data['metadatas']
                    )
                ]
                
                # Initialize SwarmRetriever (Workers)
                self.hybrid_retriever = HybridRetriever(self.retriever, docs_for_bm25)
                
                # Initialize LLMComplexityAnalyzer (Manager)
                self.complexity_analyzer = LLMComplexityAnalyzer(self.domain, self.llm)
                logger.info("✅ (Manager + Workers) initialized.")
                # --- END: NEW SWARM INITIALIZATION ---
                self.metrics_tracker = MetricsTracker(save_path=f"metrics_{self.domain}.json")
                # logger.info("✅ Metrics tracker initialized")
                # Initialize validator *after* setting domain
                self.answer_validator = AnswerValidatorAgent(self.llm, self.domain)
                 # --- This is the new, simple QA chain that will be used *after* reranking ---
                self.qa_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)

                self._initialize_agent()
                
            except Exception as e:
                logger.error(f"❌ Error during AgenticQA setup for '{self.domain}': {e}", exc_info=True)
        else:
            logger.warning("⚠️ AgenticQA initialized without a config. Retriever will be None.")

    def run_rag_with_reranking(self, query: str, chat_history: list,expanded_query: str = None,return_only_context=False) -> str:
        """
        This is worker for Manager(React).
        Enhanced Swarm-RAG pipeline with adaptive retrieval and reranking.
        
        Pipeline:
        1. Contextualize query
        2. Refine query
        3. ComplexityAnalyzer (Manager) determines optimal k
        4. HybridRetriever (Workers) deploys parallel retrievers with k
        5. Rerank combined swarm results
        6. Filter results by threshold
        7. Generate Answer
        """
        logger.info(f"--- 🐝 SWARM RAG (with Reranker) PIPELINE RUNNING for query: '{query}' ---")
        
        if not self.reranker or not self.hybrid_retriever or not self.complexity_analyzer:
            logger.error("Retriever components or Reranker not initialized. Cannot perform RAG.")
            return "Error: RAG components are not available."

        try:
            if expanded_query:
                standalone_query = expanded_query
                logger.info(f"Using EXPANDED QUERY for RAG: {standalone_query}")
            else:
                standalone_query = query
            # # 1. Contextualize query
            # standalone_query = query
            logger.info(standalone_query)
            if chat_history:
                contextualize_chain = self.contextualize_q_prompt | self.llm
                response = contextualize_chain.invoke({"chat_history": chat_history, "input": query})
                standalone_query = response.content
                logger.info(f"Contextualized query: '{standalone_query}'")
            
            # 2 - REFINE QUERY ---
            logger.info("Refining query for search...")
            response = self.refiner_chain.invoke({"query": standalone_query})
            refined_query = response.content.strip()
            logger.info(f"Refined query: '{refined_query}'")
            
            
            # 3. Complexity analysis
            analysis = self.complexity_analyzer.analyze(standalone_query)
            k = analysis['k']
            self._last_complexity_analysis = analysis
            logger.info(f"Query complexity: {analysis['complexity'].upper()} | k={k}")
            
            # 4. Retrieve with Workers
            swarm_docs = self.hybrid_retriever.retrieve_with_swarm(refined_query, k=k)
            
            if not swarm_docs:
                self._last_context = None
                logger.warning("Swarm Retriever found no documents.")
                return "I do not know the answer to that as it is not in my documents."
            
            # 5. Format for Reranker
            passages = [
                {"id": i, "text": doc.page_content, "meta": doc.metadata} 
                for i, doc in enumerate(swarm_docs)
            ]
            
            # 6. Rerank
            logger.info(f"Reranking {len(passages)} swarm-retrieved documents...")
            rerank_request = RerankRequest(query=standalone_query, passages=passages)
            reranked_results = self.reranker.rerank(rerank_request)
            
            top_score = reranked_results[0]['score'] if reranked_results else 0
            logger.info(f"Reranking complete. Top score: {top_score:.3f}")

            # 7. Filter 
            threshold = 0.1
            if self.domain == "islamic":
                threshold = 0.15 
            elif self.domain == "medical":
                threshold = 0.15
            else:
                threshold = 0.10
                
            logger.info(f"Using threshold={threshold} for {self.domain} domain")
            final_docs = []
            worker_contributions = {}
            
            for result in reranked_results:
                if result['score'] > threshold:
                    # Re-create the Document object from reranked data
                    doc = Document(
                        page_content=result['text'],
                        metadata=result.get('meta', {})
                    )
                    final_docs.append(doc)
                    
                    # Track worker contributions in final answer
                    worker = result.get('meta', {}).get('swarm_worker', 'unknown')
                    worker_contributions[worker] = \
                        worker_contributions.get(worker, 0) + 1
            
            print(f"Filtered to {len(final_docs)} documents above threshold {threshold}.")
            logger.info(f"Final doc contributions: {worker_contributions}")
            
            self.metrics_tracker.log_worker_contribution(worker_contributions)
            
            if final_docs:
                # 1. Log Metadata
                sources = [doc.metadata.get('source', 'unknown') for doc in final_docs]
                logger.info(f"Retrieved documents: {sources}")
                
                # 2. Log Context
                contexts = [doc.page_content for doc in final_docs]
                logger.info(f"Context : {contexts}")
                
                # 3. Deduplicate and Save for Answer()
                seen = set()
                deduped_lines = []
                for item in contexts:
                    if item not in seen:
                        seen.add(item)
                        deduped_lines.append(item)
                
                self._last_context = "\n".join(deduped_lines)
            else:
                self._last_context = None
                
            if return_only_context:
                # Exit early: just return the retrieved docs
                return final_docs 
            
            # 8. Respond
            if not final_docs:
                logger.warning("No documents passed the reranker threshold. Returning 'I don't know.'")
                return "I do not know the answer to that as my document search found no relevant information."
            print(f"Context: {final_docs}")
            # Call the QA chain with the *reranked, filtered* docs
            response = self.qa_chain.invoke({
                "context": final_docs,
                "chat_history": chat_history,
                "input": query,
                "domain": self.domain
            })
            
            logger.info("🐝 Swarm RAG pipeline complete. Returning answer.")
            return response

        except Exception as e:
            logger.error(f"Error in Swarm RAG pipeline: {e}", exc_info=True)
            return "An error occurred while processing your request."
        
    def _initialize_agent(self):
        """Build the ReAct agent"""
        """A helper function to build the agent components."""

        logger.info(f"Initializing agent for domain: '{self.domain}'")
        
        # Store chat_history as instance variable so tools can access it
        self._current_chat_history = []
        
        
        def rag_tool_wrapper(query: str) -> str:
            """Wrapper to pass chat history to RAG pipeline and calls run_rag_with_reranking method"""
            return self.run_rag_with_reranking(query, self._current_chat_history)
        
        self.tools = [
            Tool(
                name="RAG",
                func=rag_tool_wrapper,# Calls hybrid retriever(Worker) to retrieve context
                description=(f"Use this tool FIRST to search and answer questions about the {self.domain} domain using internal vector database.")
            )
            
        ]
        
        # --- DOMAIN-SPECIFIC TOOLS ---
        if self.domain == "insurance":
            # Add a specific tool for searching Etiqa's website
            etiqa_search_tool = TavilySearch(max_results=3)
            etiqa_search_tool.description = "Use this tool to search the Etiqa Takaful website for products NOT in the RAG context (e.g., medical, travel)."

            original_etiqa_func = etiqa_search_tool.invoke
            def etiqa_site_search(query):
                return original_etiqa_func(f"site:etiqa.com.my {query}")
            
            self.tools.append(Tool(
                name="EtiqaWebSearch",
                func=etiqa_site_search,
                description=etiqa_search_tool.description
            ))
            
            # Add a general web search tool
            self.tools.append(Tool(
                name="GeneralWebSearch",
                func=TavilySearch(max_results=2).invoke,
                description="Use this tool as a fallback for general, non-Etiqa questions (e.g., 'What is takaful?')."
            ))
        elif self.domain == "islamic":
            # Trusted Islamic sources for Malaysia
            islamic_search = TavilySearch(max_results=3)
            
            def islamic_trusted_search(query):
                # Search only trusted Malaysian Islamic authorities
                sites = "site:muftiwp.gov.my OR site:zulkiflialbakri.com OR site:IslamQA.org"
                return islamic_search.invoke(f"{sites} {query}")
            
            self.tools.append(Tool(
            name="TrustedIslamicSearch",
            func=islamic_trusted_search,
            description=(
                "Use this tool if RAG has incomplete or no answer. "
                "Searches ONLY trusted Malaysian Islamic sources: "
                "Pejabat Mufti Wilayah Persekutuan (muftiwp.gov.my) and "
                "Dr Zulkifli Mohamad Al Bakri (zulkiflialbakri.com/category/soal-jawab-agama/). "
                "These follow Shafi'i madhab which is official in Malaysia."
            )
        ))
        
            # General fallback (last resort)
            self.tools.append(Tool(
                name="GeneralWebSearch",
                func=TavilySearch(max_results=2).invoke,
                description="Last resort: Use only for general Islamic terms or definitions not found in RAG or trusted sources."
        ))
        else:
            medical_search = TavilySearch(max_results=3)

            def medical_trusted_search(query):
                sites = "(site:mayoclinic.org OR site:moh.gov.my OR site:nhs.uk OR site:myhealth.gov.my)"
                return medical_search.invoke(f"{sites} {query}")

            
            self.tools.append(Tool(
                name="TrustedMedSearch",
                func=medical_trusted_search,
                description=(
                "Invoke this tool if the RAG system does not return relevant medical documents. "
                "It searches only trusted health authorities: NHS (UK) and Mayo Clinic (US) for general guidance, "
                "and MOH (Ministry of Health Malaysia) plus MyHealth.gov.my for Malaysia-specific health information. "
            )))
            # Medical and Islamic domains only get the general web search fallback
            self.tools.append(Tool(
                name="GeneralWebSearch",
                func=TavilySearch(max_results=2).invoke,
                description="Use this tool as a fallback if the RAG tool finds no relevant information or if the query is about a general topic."
            ))
        
        agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=self.react_docstore_prompt)
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            handle_parsing_errors=True,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=7
        )
        logger.info(f"✅ Agent Executor(ReAct Agent) created successfully for '{self.domain}'.")
    
        
    def answer(self, query, chat_history=None):
        """
        Process a query using the main "Manager" agent and return a clean dictionary.
        
        This method orchestrates the full RAG pipeline:
        1.  Activates the main ReAct "Manager" agent (self.agent_executor).
        2.  The Manager agent, based on its tools, decides to call the "RAG" tool.
        3.  The "RAG" tool triggers the "Worker" pipeline (self.run_rag_with_reranking)
            to retrieve, rerank, and get the context-aware answer.
        4.  The final answer and all metadata (thoughts, source, etc.) are collected
            from the agent's run and returned as a single dictionary.

        Args:
            query (str): User's question
            chat_history (list): List of previous messages (AIMessage, HumanMessage)
            
        Returns:
            dict: A comprehensive dictionary containing:
                - 'answer' (str): The final generated answer string.
                - 'context' (str): The retrieved context used for the answer.
                - 'validation' (tuple): (bool, str) from the AnswerValidatorAgent.
                - 'source' (str): The tool or database used (e.g., "Domain Database (RAG)").
                - 'thoughts' (str): The ReAct agent's step-by-step thought process.
                - 'response_time' (float): Total time taken for the query.
                - 'complexity' (dict): The analysis from LLMComplexityAnalyzer.
        """
        if chat_history is None:
            chat_history = []
        self._current_chat_history = chat_history
        if not self.agent_executor:
            return {"answer": "Error: Agent not initialized.", "context": "", "validation": (False, "Init failed"), "source": "Error"}
        # START TIMING
        start_time = self.metrics_tracker.start_query()
        print(f"\n📝 AGENTIC_QA PROCESSING QUERY: '{query}'")
        
        # Manager agent(ReAct) wakes up and look at RAG first since it is the first tool
        response = self.agent_executor.invoke({
            "input": query,
            "chat_history": chat_history,
            "domain": self.domain, # Pass domain to agent
            "metadata": {
                "domain": self.domain
            }
        })
        thoughts= ""
        
        final_answer = response.get("output", "Could not generate an answer")
        
        tool_used = None
        if "intermediate_steps" in response:
            thought_log= []
            for step in response["intermediate_steps"]:
                # --- FIX: Unpack the (Action, Observation) tuple first ---
                action, observation = step
                
                if isinstance(action, AgentAction) and action.tool:
                    tool_used = action.tool #Capture the last tool used
                
                # Append Thought, Action, Action Input & Observation 
                thought_log.append(action.log)
                thought_log.append(f"\nObservation: {str(observation)}\n---") 
            
            thoughts = "\n".join(thought_log)     

        # Assign source based on the LAST tool used
        if tool_used == "RAG":
            source = "Etiqa Takaful Database" if self.domain == "insurance" else "Domain Database (RAG)"
        elif tool_used == "EtiqaWebSearch":
            source = "Etiqa Website Search"
        elif tool_used == "TrustedIslamicSearch":
            source = "Mufti WP & Dr Zul Search"
        elif tool_used == "GeneralWebSearch":
            source = "General Web Search"
        else:
            source = "Agent Logic"

        logger.info(f"Tool used: {tool_used}, Source determined: {source}")
        
        if (source.endswith("(RAG)") or source.startswith("Etiqa Takaful Database")) and self._last_context:
            context = self._last_context
        elif "Web" in source:
            context = "Web search results were used. See 'Observation' in thoughts log."
        else:
            context = "No RAG context retrieved."
        
        validation = self.answer_validator.validate(query, final_answer, source=source)
        # END TIMING
        response_time = self.metrics_tracker.end_query(start_time)
        
        complexity_info = getattr(self, '_last_complexity_analysis', None)
    
        # LOG METRICS
        self.metrics_tracker.log_query(
            query=query,
            domain=self.domain,
            source=source,
            complexity=complexity_info,
            validation=validation,
            response_time=response_time,
            answer_preview=final_answer
        )
        return {"answer": final_answer, "context": context, "validation": validation, "source": source, "thoughts": thoughts,"response_time": response_time,
        "complexity": complexity_info}

class AnswerValidatorAgent:
    def __init__(self, llm, domain="general"):
        self.llm = llm
        self.domain = domain
        self.general_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an answer validator. Check if the generated answer is factually correct "
                "and relevant to the query. Return 'Valid' if the answer is correct and relevant, "
                "or 'Invalid: [reason]' if not, where [reason] is a brief explanation of the issue."
            )),
            ("human", "Query: {query}\nAnswer: {answer}")
        ])
        self.medical_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an answer validator. Check if the generated answer is factually correct, "
                "relevant to the query, and consistent with known medical knowledge. "
                "Return 'Valid' if the answer is correct and relevant, or 'Invalid: [reason]' if not, "
                "where [reason] is a brief explanation of the issue. "
                "**Pay close attention to contradictions.** If an answer gives advice and then "
                "contradicts it (e.g., 'switch immediately' and then 'always consult your doctor first'), "
                "it is **Invalid** because it is unsafe and confusing."
            )),
            ("human", "Query: {query}\nAnswer: {answer}")
        ])
        self.islamic_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an answer validator for Islamic Fiqh or anything related to Islam. Check if the answer correctly addresses "
                "the query based on the provided sources. The answer should be neutral and present "
                "the required perspectives (e.g., Shafi'i and Maliki) if available. "
                "Return 'Valid' if the answer is correct and relevant, or 'Invalid: [reason]' if not."
            )),
            ("human", "Query: {query}\nAnswer: {answer}")
        ])

    def validate(self, query, answer, source="RAG"):
        if self.domain == "insurance":
            logger.info("Skipping validation for insurance domain.")
            return True, "Validation skipped for insurance domain."
        
        try:
            # --- 11. IMPROVED VALIDATOR LOGIC ---
            # Choose the right prompt based on domain and source
            prompt = self.general_prompt # Default
            if source == "RAG" or "Database" in source:
                if self.domain == "medical":
                    prompt = self.medical_prompt
                elif self.domain == "islamic":
                    prompt = self.islamic_prompt
            
            response = self.llm.invoke(prompt.format(query=query, answer=answer))
            validation = response.content.strip()
            logger.info(f"AnswerValidator result for query '{query}': {validation}")

            if validation.lower().startswith("valid"):
                return True, "Answer is valid and relevant."
            elif validation.lower().startswith("invalid"):
                reason = validation.split(":", 1)[1].strip() if ":" in validation else "No reason provided."
                return False, reason
            else:
                return False, "Validation response format unexpected."
        except Exception as e:
            logger.error(f"AnswerValidator error: {str(e)}")
            return False, "Validation failed due to error."