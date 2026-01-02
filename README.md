# ALPHA: Agentic LLM Platform Blueprint

**A modular blueprint for building agentic RAG systems — from single-agent ReAct to multi-agent swarm workflows**

## 🎯 Overview

ALPHA is a baseline agentic RAG architecture designed to be adapted for:

- 🧑‍🤝‍🧑 **Citizen-facing systems** (public Q&A, knowledge access)
- 🏢 **Organisational systems** (enterprise, healthcare, legal, internal docs)

**This repository is not a finished product.**  
It is a starting point that demonstrates design patterns, not a production system.

> **Rule of thumb:**  
> If your use case is simple → use a single agent  
> If your task requires diverse reasoning → use a multi-agent swarm

---

## 🧠 Agent Design Choices (IMPORTANT)

ALPHA supports two agentic workflows. You must choose and modify based on your industry needs.

### 🔹 Option 1: Single-Agent with ReAct (Recommended Baseline)

**Use when:**
- One agent is sufficient to reason + retrieve
- Tasks are straightforward
- You want lower cost & complexity

**File to modify:** `doc_qa.py`

**Characteristics:**
- One agent
- ReAct-style reasoning
- Deterministic and easier to control

**Suitable for:**
- Citizen services
- FAQ systems
- Internal documentation bots

### 🔹 Option 2: Multi-Agent Swarm (Diverse Reasoning)

**Use when:**
- A single agent is not enough
- You need multiple perspectives to solve one task
- The domain is complex or high-stakes

**File to modify:** `swarm_qa.py`

**Characteristics:**
- Multiple expert agents
- Parallel execution
- Aggregation / summarization agent

**Suitable for:**
- Medical reasoning
- Legal analysis
- Policy interpretation
- Research-heavy workflows

> ⚠️ **Multi-agent ≠ always better**  
> It increases cost, latency, and complexity.

---

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Trojanhammer/ALPHA-Agentic-LLM-Platform-for-Holistic-Automation.git
cd ALPHA-Agentic-LLM-Platform-for-Holistic-Automation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup environment
cp .env.example .env
# Edit .env and add your API keys

```
## 📦 Data Ingestion (MANDATORY)

⚠️ ChromaDB is empty by default  
You must ingest your own data.

**Example:**
```bash
python add_csv.py --file data/medical_dataset/medquad.csv
Available ingestion examples:

add_csv.py → structured data

add_books.py → long-form text

These scripts are examples, not production pipelines.
```

## 📁 Project Structure
```
ALPHA-Agentic-LLM-Platform-for-Holistic-Automation/
├── medical_dataset/          # Sample datasets (POC only) & Example domain
├── add_csv.py                # CSV ingestion example
├── add_books.py              # Text ingestion example
├── doc_qa.py                 # Single-agent ReAct workflow
├── swarm_qa.py               # Multi-agent swarm workflow
├── chroma_db/                # Vector database (ignored in git)
└── requirements.txt
```

## 🧩 Customisation Is REQUIRED
This blueprint does not handle:

* Images, tables, or diagrams

* OCR

* Vision-language reasoning

* Security or access control

Example:  
If your organisation ingests PDFs with charts, scanned images, or medical diagrams → you must add:

* OCR (e.g., Tesseract)

* Vision models (e.g., Gemini Vision)

* Multimodal chunking logic

## 🏗️ Intended Use Cases
Originally Designed For:

* Citizen service automation

* Public knowledge access

* Community information systems

Extended For Industry:

* Healthcare (clinical Q&A)

* Legal (case & contract analysis)

* Enterprise (SOPs, internal docs)

## ⚠️ Critical Notes
What's NOT included:

* ❌ Not production-ready

* ❌ No main entry point

* ❌ No deployment pipeline

* ❌ No monitoring or logging

You must build:

* Your own API

* Security layers

* Domain-specific logic

## 🎓 Academic Note (Final Year Project)
This repository is used as a proof-of-concept (POC) for a Final Year Project (FYP).

Public datasets are used for validation

Architecture is research-driven

Domains are examples, not limitations

The goal is to demonstrate agentic system design, not domain ownership.

## 🧠 Key Takeaway
ALPHA shows how to build agentic RAG systems, not what system to deploy.

> Start simple.
> Add complexity only when the task demands it.
