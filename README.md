# AcademiaAgent: A Local-First, Agentic RAG System for Educational Support

![Version](https://img.shields.io/badge/version-1.0.0--alpha-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Local-First](https://img.shields.io/badge/deployment-Local--First-orange)
![LLM](https://img.shields.io/badge/LLM-Ollama--Gemma4-purple)

AcademiaAgent is a production-ready prototype of a local-first intelligence layer designed specifically for academic environments. Built on a sophisticated **agentic, self-correcting RAG (Retrieval-Augmented Generation)** architecture, it provides students, faculty, and administrators with a grounded, reliable interface for navigating complex institutional policies, course materials, and research inquiries.

In high-stakes academic environments, the cost of a "hallucination" is high. A student misinformed about a graduation requirement or a researcher relying on a fake citation can face significant consequences. AcademiaAgent is engineered to solve this by prioritizing **factuality, data sovereignty, and a strict "grounding-first" philosophy.**

---

## 📖 Table of Contents
- [The Problem Space](#-the-problem-space)
  - [The RAG Gap in Academia](#the-rag-gap-in-academia)
  - [The Local-First Manifesto](#the-local-first-manifesto)
- [System Architecture](#-system-architecture)
  - [High-Level Component Map](#high-level-component-map)
  - [Data Flow Diagram](#data-flow-diagram)
- [The RAG Pipeline Deep Dive](#-the-rag-pipeline-deep-dive)
  - [Data Ingestion & Corpus Management](#data-ingestion--corpus-management)
  - [Hybrid Retrieval Logic](#hybrid-retrieval-logic)
    - [Dense Retrieval (Cosine Similarity)](#dense-retrieval-cosine-similarity)
    - [Sparse Retrieval (BM25)](#sparse-retrieval-bm25)
    - [Dynamic Alpha Scaling](#dynamic-alpha-scaling)
  - [The Reranking Stage](#the-reranking-stage)
  - [The "Lost in the Middle" Phenomenon](#the-lost-in-the-middle-phenomenon)
- [The Agentic Control Loop](#-the-agentic-control-loop)
  - [Intent Analysis & Routing](#intent-analysis--routing)
  - [The Self-Correction Mechanism (Corrective RAG)](#the-self-correction-mechanism-corrective-rag)
  - [Grounded Answer Composition](#grounded-answer-composition)
  - [The Reflection Loop](#the-reflection-loop)
- [Guardrails & Academic Integrity](#-guardrails--academic-integrity)
  - [Academic Integrity Filter](#academic-integrity-filter)
  - [Prompt Injection Mitigation](#prompt-injection-mitigation)
  - [Role-Based Access Control (RBAC)](#role-based-access-control-rbac)
- [Technical Stack](#-technical-stack)
- [Detailed Implementation Walkthrough](#-detailed-implementation-walkthrough)
  - [The Cognitive Layer (agents.py)](#the-cognitive-layer-agentspy)
  - [The Search Engine (retrieval.py)](#the-search-engine-retrievalpy)
  - [The Knowledge Base (corpus.py)](#the-knowledge-base-corpuspy)
- [Getting Started](#-getting-started)
- [Operational Guide & Tuning](#-operational-guide--tuning)
  - [Tuning the Hybrid Alpha](#tuning-the-hybrid-alpha)
  - [Customizing the Synonym Lexicon](#customizing-the-synonym-lexicon)
  - [Managing Seed Data](#managing-seed-data)
- [Production Roadmap](#-production-roadmap)
- [Ethical AI in Education](#-ethical-ai-in-education)
  - [The AI as a Tutor vs. Proxy](#the-ai-as-a-tutor-vs-proxy)
  - [FERPA & GDPR Compliance](#ferpa--gdpr-compliance)
- [FAQ & Troubleshooting](#-faq--troubleshooting)
- [Contributing](#-contributing)

---

## 🎯 The Problem Space

### The "RAG Gap" in Academia
Standard Retrieval-Augmented Generation (RAG) typically follows a linear pipeline: `Query` $\rightarrow$ `Vector Search` $\rightarrow$ `LLM Generation`. While this is sufficient for general-purpose chatbots, it creates a "precision gap" in academic contexts.

#### 1. The Brittle Nature of Policy
Academic regulations are not "conceptual"—they are literal. For example, a policy stating that a student "must complete 120 credits" is fundamentally different from "should complete 120 credits." Vector embeddings, which map words to a high-dimensional conceptual space, often treat "must" and "should" as semantically similar because they both relate to "requirements." In a university setting, this "smoothing" effect can lead to an AI giving a student false hope about their graduation eligibility.

#### 2. The Hallucinated Citation Crisis
LLMs are probabilistic engines. When asked to provide a source, they often generate a citation that *looks* correct (e.g., "Smith et al., 2021, Journal of Higher Ed") but does not actually exist. In academia, a fake citation is not just a bug; it is a breach of scholarly integrity.

#### 3. The Sovereignty Dilemma
University data is highly sensitive. Student records, faculty research in progress, and internal administrative memos are protected by laws like FERPA (US) and GDPR (EU). Sending this data to a cloud-based LLM provider—where it may be used for future training—is a non-starter for most institutional legal teams.

### The Local-First Manifesto
AcademiaAgent solves these problems by adhering to a **Local-First** architecture. By running the LLM (via Ollama) and the retrieval engine on the institution's own hardware:
- **Zero Data Leakage**: No data leaves the internal network.
- **Deterministic Control**: We can implement hard-coded guardrails that cannot be bypassed by "prompt hacking" a cloud API.
- **Customized Intelligence**: The system can be augmented with local "seed data" (handbooks, catalogs) that the general public cannot access, creating a truly bespoke institutional brain.

---

## 🏗 System Architecture

AcademiaAgent is structured as a modular pipeline where each stage is inspectable, measurable, and replaceable.

### High-Level Component Map
- **`app/main.py`**: The **Gateway**. A FastAPI implementation that provides the REST interface. It manages the request lifecycle and ensures that user roles are correctly passed through the system.
- **`app/agents.py`**: The **Cognitive Layer**. This is the "Brain" of the system. It implements the agentic loop, deciding when to rewrite queries, when to trigger guardrails, and how to compose the final grounded response.
- **`app/retrieval.py`**: The **Search Engine**. This is where the hybrid retrieval logic resides. It coordinates between the dense (cosine) and sparse (BM25) search methods.
- **`app/corpus.py`**: The **Knowledge Base**. This module manages the "physical" state of the data—how it is chunked, stored in memory, and filtered by role.
- **`app/ingestion.py`**: The **Data Pipeline**. This is the entry point for new knowledge. It handles the complexities of parsing PDFs and Markdown files into a format the retriever can understand.
- **`app/ollama.py`**: The **LLM Interface**. A specialized wrapper that configures the local model for two distinct tasks: *Query Rewriting* (creative/expansive) and *Grounded Generation* (strict/restrictive).

### Data Flow Diagram
`User Query` $\rightarrow$ `Intent Routing` $\rightarrow$ `Hybrid Retrieval` $\rightarrow$ `Quality Check` $\rightarrow$ `(Optional) Query Rewrite Loop` $\rightarrow$ `Grounded LLM Generation` $\rightarrow$ `Citation Verification` $\rightarrow$ `Final Response`.

---

## 🔍 The RAG Pipeline Deep Dive

### Data Ingestion & Corpus Management
Knowledge is only as good as its representation. AcademiaAgent avoids "naive chunking" (splitting every 500 characters), which often cuts a policy rule in half.

#### Structural Parsing
The ingestion pipeline identifies headers (`#`, `##`) and list markers. It attempts to keep "logical units" together. If a section is titled "Graduation Requirements," the system ensures that the requirements list remains a single chunk, preserving the relationship between the header and the bullet points.

#### Role-Based Metadata
Every chunk is assigned a `role_access` level. 
- **Public**: Accessible to everyone.
- **Student**: Accessible to students and above.
- **Faculty**: Accessible to faculty and admins.
- **Admin**: Restricted to high-level administration.

When a query is processed, the `AcademicCorpus` filters the search space to only include chunks that match the user's role, preventing unauthorized access to sensitive internal memos.

### Hybrid Retrieval Logic
To solve the "precision vs. recall" trade-off, AcademiaAgent employs a hybrid system.

#### 1. Dense Retrieval (Cosine Similarity)
Dense retrieval treats documents as vectors in a high-dimensional space. It is excellent for **Recall**—finding documents that are conceptually related even if they don't share the same keywords.
For example, a query for "financial assistance" will successfully retrieve documents mentioning "scholarships," "grants," and "bursaries."

**The Math:**
The similarity is the cosine of the angle between the query vector $q$ and the document vector $d$:
$$\text{similarity} = \frac{\sum_{i=1}^{n} q_i d_i}{\sqrt{\sum_{i=1}^{n} q_i^2} \sqrt{\sum_{i=1}^{n} d_i^2}}$$

#### 2. Sparse Retrieval (BM25)
Sparse retrieval is the gold standard for **Precision**. It treats documents as "bags of words." If a user searches for `CS101`, they don't want a "semantically similar" course like `CS102`; they want the exact document for `CS101`. 

AcademiaAgent implements **BM25 (Best Matching 25)**, which improves upon TF-IDF by introducing:
- **Term Frequency Saturation**: The benefit of a word appearing multiple times in a document diminishes quickly.
- **Document Length Normalization**: Shorter documents that contain the keyword are ranked higher than massive documents that contain the keyword only once.

#### 3. Dynamic Alpha Scaling
The "secret sauce" of the retriever is the **Dynamic Alpha ($\alpha$)**. Instead of a static blend, the system analyzes the query:
- **Exact Match Trigger**: If the query contains a course code (e.g., `BIO202`) or a strong policy keyword (e.g., `deadline`), the system sets $\alpha$ to a low value (e.g., $0.35$). This shifts the weight toward **BM25**, ensuring precision.
- **Conceptual Trigger**: If the query is a broad question (e.g., "How do I balance study and life?"), the system sets $\alpha$ to a higher value (e.g., $0.72$), favoring **Cosine Similarity** for better conceptual recall.

### The Reranking Stage
Hybrid search produces a "candidate list." To ensure the LLM receives the most relevant information, the system performs a final **Reranking Pass**.

The reranker evaluates **Term Coverage**. It asks: *"Of the key terms in the user's query, how many are actually present in this chunk?"* Documents with $100\%$ coverage of the query's core terms are boosted to the top, regardless of their hybrid score. This effectively filters out "false positives" that might have had a high cosine similarity but lacked the specific facts required to answer the question.

### The "Lost in the Middle" Phenomenon
Recent research shows that LLMs struggle to extract information located in the middle of a long context window—they remember the beginning and the end best. AcademiaAgent mitigates this by:
1. **Strict Top-K Filtering**: Only providing the top 5 most relevant chunks.
2. **Reranking**: Ensuring the absolute most critical information is placed at the very top (Position 1) and the very bottom (Position 5) of the prompt, maximizing the model's attention.

---

## 🤖 The Agentic Control Loop

AcademiaAgent is not a passive pipeline; it is an active agent that reasons about its own performance.

### Intent Analysis & Routing
The `AcademicAgent` first determines the *intent* of the query. This is not just for organization; it changes the **system prompt** used by the LLM.
- **Policy Route**: The LLM is told to be "Formal, literal, and cautious."
- **Pedagogy Route**: The LLM is told to be "Encouraging, explanatory, and Socratic."
- **Research Route**: The LLM is told to "Prioritize synthesis and scholarly tone."
- **Integrity Route**: The system bypasses the LLM generation and returns a hard-coded refusal to produce assessed work.

### The Self-Correction Mechanism (Corrective RAG)
This is the core innovation of the system. Instead of blindly trusting the first retrieval, the agent performs a **Quality Check**.

**The Loop Logic:**
1. **Retrieve**: Get the top chunks.
2. **Score**: Calculate the average `rerank_score` of the top 3 hits.
3. **Evaluate**: If the score is $< 0.20$, the agent concludes: *"The information I found is too weak to provide a reliable answer."*
4. **Rewrite**: The agent uses the LLM to rewrite the query.
   - *Original*: "How do I get money for school?"
   - *Rewritten*: "What are the institutional financial aid applications, scholarship eligibility criteria, and tuition payment deadlines?"
5. **Repeat**: The agent re-runs the retrieval with the expanded query.
6. **Final Verdict**: If after 3 attempts the score remains low, the system admits defeat. This is a "Feature, not a Bug"—it is better for an academic AI to say "I don't know" than to invent a policy.

### Grounded Answer Composition
Once context is secured, the agent generates the answer using a **Strict Grounding Template**.

The prompt forces the model to follow this structure:
`[Claim] -> [Citation Index]`.

Example: *"Students must maintain a 2.0 GPA to avoid probation [1]. All applications for the Dean's scholarship must be submitted by March 1st [3]."*

### The Reflection Loop
After the LLM generates a response, the system performs a **Final Reflection Pass**. It scans the response for any claims that *lack* a citation. If the LLM says "You should check with the registrar" without a citation to the registrar's office policy, the system flags the response as "Ungrounded" and triggers a fallback message.

---

## 🛡 Guardrails & Academic Integrity

### Academic Integrity Filter
The `Integrity Route` uses a combination of regex and keyword analysis to detect "cheating requests." 
Patterns like `(write|solve|complete) + (my|this) + (assignment|exam)` trigger an immediate block. However, the agent is programmed to be a **Tutor**. Instead of a cold "No," it says: *"I cannot write your essay, but I can help you brainstorm an outline or explain the core concepts of the prompt."*

### Prompt Injection Mitigation
Prompt injection (e.g., "Ignore all previous instructions") is a major risk for RAG systems. AcademiaAgent implements a **Pre-processing Shield**. Any query containing injection patterns is intercepted *before* it ever reaches the LLM, and a standard "Safeguard" message is returned.

### Role-Based Access Control (RBAC)
RBAC is integrated at the **Corpus Level**, not the API level. This means that even if the LLM wanted to "leak" a secret faculty memo, it wouldn't be able to—because that memo was never retrieved from the database in the first place. The retriever only sees what the user's role is permitted to see.

---

## 💻 Technical Stack

| Component | Technology | Justification |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Native support for AI libraries and fast prototyping. |
| **API** | FastAPI | Asynchronous request handling for low-latency RAG. |
| **LLM** | Ollama / Gemma 4 | High-reasoning local model that fits on consumer GPUs. |
| **Search** | Hybrid (BM25 + Cosine) | Bridges the gap between conceptual search and literal precision. |
| **Parsing** | PyPDF2 / Markdown | Handles the most common academic document formats. |
| **Container** | Docker | Ensures "one-click" deployment across different university servers. |

---

## 🛠 Detailed Implementation Walkthrough

### The Cognitive Layer (`agents.py`)
The `AcademicAgent` class acts as a state machine. The `answer()` method is the entry point. It orchestrates the `route_query` $\rightarrow$ `retriever.search` $\rightarrow$ `ollama.rewrite_query` $\rightarrow$ `ollama.grounded_answer` sequence. The use of a `RetrievalTrace` object allows every decision—from the `hybrid_alpha` used to the `loop_count`—to be logged and displayed to the user, making the AI's "thought process" transparent.

### The Search Engine (`retrieval.py`)
The `HybridRetriever` class is designed to be "Adapter-ready." While it currently uses an in-memory `AcademicCorpus`, the methods `_cosine` and `_bm25` are decoupled from the data storage. This means swapping the internal list for a **Qdrant** or **ElasticSearch** cluster requires changing only the `search()` method, leaving the agent logic untouched.

### The Knowledge Base (`corpus.py`)
The `AcademicCorpus` handles the "Heavy Lifting" of data management. It implements the `chunks_for(role)` method, which performs the RBAC filtering. It also manages the mapping between `chunk_id` and `document_id`, allowing the system to provide full document titles and section headers in the final citations.

---

## 🚀 Getting Started

### Prerequisites
- **Ollama** installed and running on your local machine.
- **Gemma 4 (12b)** model: `ollama pull gemma4:12b`.
- **Python 3.10+**.

### Installation
```bash
# 1. Clone and enter
git clone https://github.com/Shreyan-saxena/Academia-Agent
cd Academia-Agent

# 2. Setup Environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install with document support
pip install -e .[dev,documents]

# 4. Run the API
uvicorn app.main:app --reload
```

### Basic Usage
1. Navigate to `http://localhost:8000`.
2. Use the **Upload** tab to add a PDF of your course syllabus.
3. Ask: *"What is the late submission policy for the final project?"*
4. Observe the **Live Trace** to see the agent identify the `Policy Route`, perform a hybrid search, and generate a grounded answer with citations.

---

## ⚙️ Operational Guide & Tuning

### Tuning the Hybrid Alpha
The `alpha` parameter in `retrieval.py` controls the balance between Dense (Semantic) and Sparse (Keyword) search.
- **Increase Alpha ($\uparrow \alpha$)**: If the system is missing conceptually related documents (e.g., it doesn't realize "financial aid" and "scholarships" are related).
- **Decrease Alpha ($\downarrow \alpha$)**: If the system is returning "similar" but incorrect documents (e.g., it returns `CS102` when the user asked for `CS101`).

### Customizing the Synonym Lexicon
The `SYNONYMS` dictionary in `retrieval.py` is a powerful tool for improving recall. For different disciplines, you should expand this list:
- **For Law**: Add `{"statute": "law regulation act", "precedent": "case law ruling"}`.
- **For Medicine**: Add `{"diagnosis": "identification symptom condition", "treatment": "therapy cure intervention"}`.

### Managing Seed Data
The `data/seed` folder contains the baseline knowledge. To add a new institutional handbook:
1. Place the `.md` or `.pdf` file in `data/seed`.
2. Restart the server.
3. The `AcademicCorpus` will automatically ingest these files on startup, making them available for retrieval.

---

## 🗺 Production Roadmap

To move AcademiaAgent from a prototype to an institution-wide deployment, we propose the following architectural upgrades:

### 1. Distributed Vector Storage
Replace the in-memory list with **Qdrant**. Qdrant's "Payload Filtering" allows us to move RBAC from Python code into the database engine itself, drastically increasing performance for million-document corpora.

### 2. Advanced Embedding Models
Swap the current TF-Cosine logic for **BGE-M3**. BGE-M3 is a "Multi-lingual, Multi-functional, Multi-granularity" embedding model that handles long documents (up to 8192 tokens) far better than standard models.

### 3. Cross-Encoder Reranking
Implement a `cross-encoder/ms-marco-MiniL-L-6-v2` model. Unlike our current "Term Coverage" reranker, a Cross-Encoder looks at the query and the document *simultaneously*, providing a much more accurate relevance score.

### 4. Enterprise Identity (SSO)
Integrate with **Azure AD** or **Okta**. This allows the `UserRole` to be automatically determined by the user's institutional login, removing the need for manual role selection in the UI.

---

## ⚖️ Ethical AI in Education

### The AI as a Tutor vs. Proxy
The fundamental tension in EdTech AI is the difference between *assistance* and *automation*. 
- **Automation (The Proxy)**: An AI that writes an essay for a student. This destroys the learning process and violates academic integrity.
- **Assistance (The Tutor)**: An AI that explains the rubric, helps a student find a source, and asks the student "What do you think about this paragraph?" 

AcademiaAgent is hard-coded to be an **Assistance** tool. By refusing to produce final deliverables, it forces the student to remain the "Human in the Loop."

### FERPA & GDPR Compliance
By employing a **Local-First** architecture, AcademiaAgent eliminates the most significant compliance risk: the "Third-Party Processor." Under GDPR, sending personal data to a US-based cloud provider often requires complex "Standard Contractual Clauses." By keeping the data on a university-owned server in the university's own jurisdiction, these legal hurdles are bypassed entirely.

---

## ❓ FAQ & Troubleshooting

### Frequently Asked Questions

**Q: Why is the agent refusing to answer my homework question?**
**A:** AcademiaAgent is designed as an educational tutor. It will refuse any request that looks like a request for a final, submit-ready answer. However, if you share your current attempt or ask for an explanation of the concept, the agent will provide targeted guidance to help you solve it yourself.

**Q: How does the system handle multiple documents with the same title?**
**A:** The `AcademicCorpus` assigns a unique `document_id` to every file upon ingestion. Citations are linked to these unique IDs, ensuring that even if you have two files named `Syllabus.pdf`, the system knows exactly which one provided the information.

**Q: Can I use a different model than Gemma 4?**
**A:** Yes. As long as the model is available via Ollama, you can change the model ID in `app/ollama.py`. We recommend models with at least 7B parameters for the best grounded reasoning.

**Q: Does the system support multi-language documents?**
**A:** The current hybrid retriever is optimized for English. For multi-language support, we recommend implementing the **BGE-M3** embedding model as outlined in the Production Roadmap.

### Troubleshooting

**Issue: The AI says it cannot find any relevant information, but I know the document is there.**
- **Check Role**: Ensure you are querying with the correct role (e.g., if the document is tagged "Faculty," a "Student" role query will not find it).
- **Check Formatting**: Ensure the PDF is not a "scanned image" PDF. The system requires selectable text. If you have a scanned PDF, please run it through an OCR tool first.
- **Tuning**: Try reducing the `alpha` parameter in `retrieval.py` to prioritize keyword matching.

**Issue: The responses are too long or too short.**
- **Tuning**: You can adjust the `top_k` parameter in the `QueryRequest` to control how many chunks are fed into the LLM. Fewer chunks result in more concise answers.

**Issue: Ollama is running slowly.**
- **Hardware**: Ensure you have sufficient VRAM. If the model is offloading to CPU, response times will increase significantly.
- **Model Choice**: Try a smaller model (e.g., `phi3` or `mistral`) to see if performance improves.

---

## 🤝 Contributing

### Project Structure
- `app/`: All core logic.
- `data/`: Knowledge base and uploads.
- `static/`: Frontend (HTML/JS/CSS).
- `tests/`: Python test suites.

### How to Add a New Route
1. **Define**: Add a new entry to the `AgentRoute` enum in `app/schemas.py`.
2. **Detect**: Update `route_query` in `app/agents.py` with the keywords that should trigger this route.
3. **Style**: Add a new "Opening" string to the `AcademicAgent.answer` method to set the tone for the response.

### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Built with 🛠️ [Ollama](https://ollama.ai/), [FastAPI](https://fastapi.tiangolo.com/), and [Python](https://www.python.org/).*
