# ============================================================
# agent.py — AI Agent Module (LLM Integration)
# ============================================================
# Responsibilities:
#   - Connect to Groq API (free, fast LLM inference)
#   - Implement RAG (Retrieval Augmented Generation) pipeline
#   - Provide specialized prompts for different research tasks:
#       1. ask_question()       → general Q&A with citations
#       2. find_research_gaps() → identify gaps and thesis topics
#       3. summarize_papers()   → structured paper summaries
#       4. generate_citation()  → APA/MLA citation formatting
#       5. compare_papers()     → cross-paper comparison
#
# What is RAG?
#   Retrieval Augmented Generation = search relevant chunks first,
#   then send them as context to the LLM. This grounds the LLM's
#   answers in YOUR papers rather than generic training data.
#
# LLM Used: Llama 3 (8B) via Groq — FREE tier, extremely fast (~1s)
#
# Author: Research Paper Agent
# ============================================================

import os
from groq import Groq
from dotenv import load_dotenv
from vectorstore import search

# Load environment variables from .env file
# This reads GROQ_API_KEY without hardcoding secrets in code
load_dotenv()

# ── Groq Client Initialization ────────────────────────────────────────────────
# Initialize once at module level — reused across all function calls
# Groq provides free, ultra-fast inference for Llama 3 models
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Model Configuration ───────────────────────────────────────────────────────
# llama3-8b-8192: 8 billion parameter model, 8192 token context window
# Fast enough for real-time streaming, smart enough for research tasks
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MAX_TOKENS = 1500


def _call_llm(prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """
    Internal helper: Send a prompt to the Groq LLM and return the response.

    Centralizes all LLM calls so model/parameters can be changed in one place.

    Args:
        prompt (str): The full prompt to send to the model.
        max_tokens (int): Maximum tokens in the response. Default 1500.

    Returns:
        str: The model's text response.

    Raises:
        Exception: If the API call fails (network error, invalid key, etc.)
    """
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {
                # System message sets the AI's role and behavior
                "role": "system",
                "content": (
                    "You are an expert research assistant specializing in academic "
                    "papers and scientific literature. You provide accurate, "
                    "well-structured, and insightful analysis. Always cite the "
                    "source paper when referencing specific information."
                )
            },
            {
                # User message contains the actual task + retrieved context
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=max_tokens,
        temperature=0.3,  # Lower = more factual/consistent (0=deterministic, 1=creative)
    )

    return response.choices[0].message.content


def _build_context(collection, query: str, n_results: int = 6) -> tuple[str, list]:
    """
    Internal helper: Retrieve relevant chunks and format them as context.

    Searches the vector store for chunks most similar to the query,
    then formats them with source attribution for the LLM prompt.

    Args:
        collection: ChromaDB collection to search.
        query (str): The search query (usually the user's question).
        n_results (int): Number of chunks to retrieve.

    Returns:
        tuple: (formatted_context_string, list_of_sources)
    """
    results = search(collection, query, n_results=n_results)

    # results['documents'] is a list of lists (one list per query)
    # We only have one query so take index [0]
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    # Format context with source labels so LLM knows which paper each chunk is from
    context_parts = []
    sources = set()

    for doc, meta in zip(documents, metadatas):
        source = meta.get("source", "Unknown Paper")
        sources.add(source)
        # Label each chunk with its paper source
        context_parts.append(f"[From: {source}]\n{doc}")

    context = "\n\n---\n\n".join(context_parts)

    return context, list(sources)


# ── Public Agent Functions ────────────────────────────────────────────────────

def ask_question(collection, question: str) -> str:
    """
    Answer a research question using RAG over uploaded papers.

    Pipeline:
        1. Embed the question
        2. Retrieve top-6 most relevant chunks from the papers
        3. Send chunks + question to LLM
        4. Return grounded answer with citations

    Args:
        collection: ChromaDB collection containing paper chunks.
        question (str): Natural language question from the user.

    Returns:
        str: LLM-generated answer grounded in the paper content.
    """
    # Step 1 & 2: Retrieve relevant context
    context, sources = _build_context(collection, question, n_results=6)

    # Step 3: Build the RAG prompt
    prompt = f"""Based on the following excerpts from research papers, answer the question accurately.
If the answer is not found in the provided context, say so clearly rather than guessing.
Always cite which paper(s) your answer comes from.

RESEARCH PAPER CONTEXT:
{context}

QUESTION: {question}

ANSWER (with citations):"""

    # Step 4: Get LLM response
    return _call_llm(prompt, max_tokens=1000)


def find_research_gaps(collection) -> str:
    """
    Analyze uploaded papers to identify research gaps and thesis opportunities.

    Uses a specialized prompt to extract:
        - Explicit limitations stated by the authors
        - Unstated gaps implied by the research scope
        - Future work suggestions
        - Potential thesis/research directions

    This is especially useful for Awais's HIT Master's research planning!

    Args:
        collection: ChromaDB collection containing paper chunks.

    Returns:
        str: Structured analysis of research gaps and opportunities.
    """
    # Search for limitation and future work sections specifically
    gap_query = (
        "limitations future work challenges open problems "
        "research gaps unsolved issues directions"
    )
    context, sources = _build_context(collection, gap_query, n_results=10)

    prompt = f"""You are a senior research advisor helping a Master's student identify research opportunities.
Analyze the following excerpts from research papers and provide a comprehensive gap analysis.

RESEARCH PAPER EXCERPTS:
{context}

PAPERS ANALYZED: {', '.join(sources)}

Please provide a structured analysis covering:

## 1. Explicit Research Gaps
(Limitations and gaps directly stated by the authors)

## 2. Implicit Research Opportunities  
(Gaps implied by the scope of the work but not explicitly stated)

## 3. Future Research Directions
(Suggestions for follow-up research based on these papers)

## 4. Potential Thesis Topics
(Specific, actionable research topics a Master's student could pursue)

## 5. Most Promising Direction
(Your top recommendation with justification)

Be specific, actionable, and grounded in the paper content."""

    return _call_llm(prompt, max_tokens=2000)


def summarize_papers(collection) -> str:
    """
    Generate structured summaries of all uploaded research papers.

    Creates a comprehensive overview including:
        - Research objectives and contributions
        - Methodology used
        - Key findings and results
        - How papers relate to each other

    Args:
        collection: ChromaDB collection containing paper chunks.

    Returns:
        str: Structured multi-paper summary.
    """
    # Broad query to capture introductions, abstracts, and conclusions
    summary_query = (
        "abstract introduction contribution methodology results "
        "conclusion findings proposed approach"
    )
    context, sources = _build_context(collection, summary_query, n_results=10)

    prompt = f"""You are an expert research summarizer. 
Analyze the following excerpts from research papers and provide comprehensive structured summaries.

RESEARCH PAPER EXCERPTS:
{context}

PAPERS FOUND: {', '.join(sources)}

For each paper identified, provide:

## Paper: [Paper Name/Title]

**Research Objective:** What problem does this paper solve?

**Key Contributions:** What are the main novel contributions?

**Methodology:** What approach/method is used?

**Key Findings:** What are the most important results?

**Significance:** Why does this work matter?

---

After summarizing individual papers, add:

## Cross-Paper Analysis

**Common Themes:** What themes appear across multiple papers?

**Methodological Comparison:** How do the approaches differ?

**Collective Contribution:** What does this body of work collectively advance?"""

    return _call_llm(prompt, max_tokens=2000)


def compare_papers(collection, aspect: str = "methodology") -> str:
    """
    Compare multiple papers on a specific aspect.

    Useful for understanding how different papers approach the same problem
    from different angles — valuable for literature review writing.

    Args:
        collection: ChromaDB collection containing paper chunks.
        aspect (str): What to compare. Options: methodology, results,
                      datasets, limitations, assumptions. Default: methodology.

    Returns:
        str: Comparative analysis across papers.
    """
    context, sources = _build_context(
        collection,
        f"compare {aspect} approach technique method dataset evaluation",
        n_results=10
    )

    prompt = f"""You are a research analyst performing a comparative literature review.
Compare the following research papers specifically focusing on their {aspect}.

RESEARCH EXCERPTS:
{context}

PAPERS: {', '.join(sources)}

Create a structured comparison of {aspect} across these papers:

## Comparison Table (describe in text)
For each paper, describe its {aspect} approach.

## Key Similarities
What do these papers have in common regarding {aspect}?

## Key Differences  
How do they differ in their {aspect}?

## Strengths and Weaknesses
For each approach, what are the trade-offs?

## Best Practice Recommendation
Based on this comparison, what approach is most suitable for what use case?"""

    return _call_llm(prompt, max_tokens=1500)


def generate_citation(paper_name: str, collection) -> str:
    """
    Generate APA and IEEE citations for an uploaded paper.

    Searches the paper's content to extract author names, year,
    title, and other metadata needed for citations.

    Args:
        paper_name (str): Filename of the paper.
        collection: ChromaDB collection to search metadata from.

    Returns:
        str: Formatted citations in APA and IEEE styles.
    """
    # Search for metadata-rich content (title page, headers, references)
    context, _ = _build_context(
        collection,
        f"author title year journal conference published {paper_name}",
        n_results=4
    )

    prompt = f"""Extract citation information from the following paper excerpts and generate citations.

PAPER: {paper_name}

PAPER EXCERPTS:
{context}

Please provide:

## Extracted Metadata
- Title: 
- Authors: 
- Year: 
- Journal/Conference: 
- Volume/Issue/Pages (if available):

## APA Citation
(Author, A. A., & Author, B. B. (Year). Title. Journal, Volume(Issue), Pages.)

## IEEE Citation
([N] A. Author, "Title," Journal, vol. X, no. X, pp. X-X, Year.)

## BibTeX Entry
(For LaTeX users)

Note: If exact metadata cannot be found in the excerpts, indicate which fields are uncertain."""

    return _call_llm(prompt, max_tokens=800)
