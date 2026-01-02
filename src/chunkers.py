from langchain_core.prompts import ChatPromptTemplate
from typing import List
from pydantic import Field,BaseModel
from langchain_classic import hub
import time, logging, uuid6
from langchain_core.documents import Document
from dotenv import find_dotenv, load_dotenv
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings, 
    ChatGoogleGenerativeAI)
from langchain_experimental.text_splitter import SemanticChunker
import re


load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Chunker:
    def __init__(self, strategy):
        self.semantic_chunker= SemanticChunker_langchain()
        self.agentic_chunker= AgenticChunker()
        self.strategy_chunker_map= {
            "semantic": self.semantic_chunker,
            "agentic": self.agentic_chunker
        }
        self.chunker= self.strategy_chunker_map[strategy]

    def build_chunks(self, docs, source):
        if self.chunker:
            return self.chunker.build_chunks(docs, source)
        else:
            logger.error("No chunker selected")
            return []

class SemanticChunker_langchain:
    # https://python.langchain.com/v0.2/docs/how_to/semantic-chunker/
    def __init__(self):
        self.embed_model_name = "models/text-embedding-004"

    def build_chunks(self, texts, source):
        # Extract raw text from Document objects if they are Documents
        raw_texts = [text.page_content if hasattr(text, 'page_content') else str(text) for text in texts]
        
        text_splitter = SemanticChunker(
            GoogleGenerativeAIEmbeddings(
                model=self.embed_model_name))

        chunks = text_splitter.create_documents(
            texts=raw_texts,
            metadatas=[{"source": source}] * len(raw_texts)
        )
        logger.info("use semantic chunk")
        return chunks

class ChunkMeta(BaseModel):
    title: str = Field(description="The title of the chunk.")
    summary: str = Field(description="The summary of the chunk.")

class ChunkID(BaseModel):
    chunk_id: int = Field(description="The chunk id.")

class Sentences(BaseModel):
    sentences: List[str]

class AgenticChunker:
    def __init__(self):
        """
        Initializes the AgenticChunker with:
        - An empty dictionary for storing chunks.
        - A large language model (LLM) for processing and summarizing text.
        - A placeholder for raw text input.
        """
        self.chunks = {}
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0
        )

        # self.raw_text = ""

    @staticmethod
    def retry_with_delay(func, *args, delay=2, retries=30, **kwargs):
        """
        Helper method to retry a function call with a delay.
        """
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(delay)
        raise RuntimeError("Exceeded maximum retries.")

    def _extract_propositions_from_page(self, page_text: str) -> List[str]:
        """
        Extracts a list of propositions from the raw text using an LLM.
        Returns the list of propositions for one page/document part.
        """
        logger.info("Extracting propositions from raw text.")
        extraction_llm = self.llm.with_structured_output(Sentences)
        obj = hub.pull("wfh/proposal-indexing")
        extraction_chain = obj | extraction_llm
    
        # CRITICAL: This now processes a SMALLER text chunk (one page)
        return self.retry_with_delay(extraction_chain.invoke, page_text).sentences

    def build_chunks(self, texts: List[Document], source=""):
        """
        Processes the list of propositions and organizes them into chunks.
        """
        chunks_as_documents=[]
        logger.info("Building chunks from propositions.")
        all_propositions =[]
        
        for i, doc in enumerate(texts):
            page_text = doc.page_content
            logger.info(f"Extracting propositions for Document/Page {i+1}...")
            
            # Call the modified extractor that returns propositions for one page
            page_propositions = self._extract_propositions_from_page(page_text)
            all_propositions.extend(page_propositions)
        
        # 2. Iterate through ALL propositions (from ALL pages) and allocate to chunks
        for proposition in all_propositions:
            self.find_chunk_and_push_proposition(proposition)
        
        # 3. Assemble final LangChain Documents
        for chunk_id in self.chunks:
            chunk_content= " ".join(self.chunks[chunk_id]["propositions"])
            chunks_as_documents.append(Document(
                page_content=chunk_content,
                metadata={"source": f"{source}_{chunk_id}"},
                id= str(uuid6.uuid6()),
                ))

        return chunks_as_documents

    def create_prompt_template(self, messages):
        """
        Helper method to create prompt templates.
        """
        return ChatPromptTemplate.from_messages(messages)

    def upsert_chunk(self, chunk_id, propositions):
        """
        Creates or updates a chunk with the given propositions.
        """
        summary_llm = self.llm.with_structured_output(ChunkMeta)
        prompt = self.create_prompt_template([
            ("system", "Generate a new or updated summary and title based on the propositions."),
            ("user", "propositions:{propositions}")
        ])
        summary_chain = prompt | summary_llm

        # Call the chain with retries
        chunk_meta = None
        try:
            chunk_meta = self.retry_with_delay(summary_chain.invoke, {"propositions": propositions})
        except RuntimeError as e:
            logger.error(f"Failed to get chunk summary after retries for ID {chunk_id}: {e}")
            # Proceed with fallback metadata to prevent crash
            
        # CRITICAL CHECK: Ensure chunk_meta is valid before accessing its attributes
        if chunk_meta and hasattr(chunk_meta, 'summary') and hasattr(chunk_meta, 'title'):
            self.chunks[chunk_id] = {
                "summary": chunk_meta.summary,
                "title": chunk_meta.title,
                "propositions": propositions
            }
        else:
            # FALLBACK: Create a basic chunk if LLM structured output fails
            logger.warning(f"LLM structured output failed for Chunk ID {chunk_id}. Using fallback summary.")
            self.chunks[chunk_id] = {
                "summary": "No LLM summary available. Content is: " + " ".join(propositions)[:100] + "...",
                "title": f"Chunk {chunk_id} (Fallback)",
                "propositions": propositions
            }

    def find_chunk_and_push_proposition(self, proposition):
        """
        Finds the most relevant chunk for a proposition or creates a new one if none match.
        """
        logger.info(f"Finding chunk for proposition: {proposition}")
        allocation_llm = self.llm.with_structured_output(ChunkID)
        allocation_prompt = self.create_prompt_template([
            ("system", "Using the chunk IDs and summaries, determine the best chunk for the proposition. "
                      "If no chunk matches, generate a new chunk ID. Return only the chunk ID."),
            ("user", "proposition:{proposition}\nchunks_summaries:{chunks_summaries}")
        ])
        allocation_chain = allocation_prompt | allocation_llm

        chunks_summaries = {
            chunk_id: chunk["summary"] for chunk_id, chunk in self.chunks.items()
        }

        best_chunk_id = self.retry_with_delay(
            allocation_chain.invoke, {
                "proposition": proposition,
                "chunks_summaries": chunks_summaries
            }
        ).chunk_id

        if best_chunk_id not in self.chunks:
            logger.info(f"Creating new chunk for proposition: {proposition}")
            self.upsert_chunk(best_chunk_id, [proposition])
        else:
            logger.info(f"Adding proposition to existing chunk ID: {best_chunk_id}")
            current_propositions = self.chunks[best_chunk_id]["propositions"]
            self.upsert_chunk(best_chunk_id, current_propositions + [proposition])
