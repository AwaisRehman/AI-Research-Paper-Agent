# 🔬 Research Paper Agent — AI-Powered Academic Research Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3%2070B-orange)

A Retrieval-Augmented Generation (RAG) tool that lets you upload PDF research papers and ask natural-language questions, find research gaps, generate structured summaries, compare papers, and produce citations — all grounded in your own documents and built entirely on free-tier tools.

## 📌 Research Paper

This project accompanies the paper:

> "Research Paper Agent: A Retrieval-Augmented Generation Tool for Academic Literature Review and Research Gap Discovery"

*(update this line once you've submitted — see the citation section below)*

## ✨ Features

- 💬 **AI Q&A** — Ask natural-language questions about your uploaded papers, with answers cited back to the source PDF, powered by Groq LLaMA 3.3 70B
- 🔍 **Research Gap Analysis** — Automatically surfaces limitations, open problems, and thesis topic opportunities
- 📝 **Structured Summaries** — Per-paper summaries plus a cross-paper synthesis
- 🔄 **Paper Comparison** — Compares uploaded papers on methodology, results, datasets, or limitations
- 📎 **Citation Generator** — Produces APA, IEEE, and BibTeX citations from extracted metadata
- 🧠 **Grounded Answers** — The AI is instructed to say "not found in the documents" rather than guessing, keeping answers tied to your sources

## 🏗️ System Architecture

```
┌──────────────────┐      uploads PDFs      ┌──────────────────┐
│   Streamlit UI    │ ─────────────────────► │   parser.py      │
│   (app.py)        │                        │   text + chunks  │
└─────────┬─────────┘                        └────────┬─────────┘
          │                                            │
          │ chat / queries                   embeds chunks (local)
          ▼                                            ▼
┌──────────────────┐                        ┌──────────────────┐
│   agent.py       │ ◄───── retrieves ────── │  vectorstore.py  │
│   Groq LLaMA 3.3 │       top-k chunks      │  ChromaDB (HNSW) │
│   70B (RAG)      │                         └──────────────────┘
└──────────────────┘
```

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF Parsing | PyMuPDF (`fitz`) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (free, local, 384-dim) |
| Vector DB | ChromaDB — in-memory, cosine similarity (HNSW) |
| LLM | Groq API — LLaMA 3.3 70B Versatile (free tier) |
| Config | python-dotenv |
| Language | Python 3.10+ |

## 📁 Project Structure

```
research-agent/
├── app.py            # Streamlit entry point — UI, tabs, session state
├── parser.py         # PDF text extraction, cleaning, chunking
├── embedder.py       # Sentence-transformer embedding generation
├── vectorstore.py    # ChromaDB collection management & similarity search
├── agent.py          # Groq LLM calls — Q&A, gap analysis, summaries, comparison, citations
├── requirements.txt  # Python dependencies
├── .env.example       # Environment variable template
├── LICENSE            # MIT License
└── README.md
```

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- A free Groq API key — [console.groq.com](https://console.groq.com)

### 1. Clone the repository
```bash
git clone https://github.com/AwaisRehman/AI-Research-Paper-Agent.git
cd AI-Research-Paper-Agent
```

### 2. Set up a virtual environment
```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your API key
```bash
cp .env.example .env
```
Then edit `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run
```bash
streamlit run app.py
```

### 6. Open in browser
[http://localhost:8501](http://localhost:8501)

## 🤖 AI Capabilities

The AI agent (Groq LLaMA 3.3 70B) retrieves relevant chunks from your uploaded papers via ChromaDB before answering, so it can handle questions like:

| Category | Example Question |
|---|---|
| Q&A | "What dataset does paper X use for evaluation?" |
| Research Gaps | "What limitations or open problems exist across these papers?" |
| Summarization | "Summarize all uploaded papers and how they relate to each other" |
| Comparison | "Compare these papers' methodologies" |
| Citation | "Generate an APA and BibTeX citation for this paper" |

## 🔧 How It Works (RAG Pipeline)

```
PDF upload → extract text (PyMuPDF) → clean & chunk (500 words, 50-word overlap)
   → embed chunks locally (MiniLM) → store in ChromaDB
   → user question → embed question → retrieve nearest chunks
   → send chunks + question to Groq (LLaMA 3.3) → grounded, cited answer
```

## 🔒 Security

- API keys loaded from `.env`, never hardcoded
- `.env` excluded from the repository via `.gitignore`
- No data persisted outside the local session — ChromaDB runs in-memory and clears on restart
- No third-party data sharing beyond the Groq API call itself

## ⚠️ Troubleshooting

| Issue | Fix |
|---|---|
| `GROQ_API_KEY not found` | Confirm `.env` exists in the project root and contains the key |
| `streamlit: command not found` | Run `python -m streamlit run app.py` instead |
| "Could not extract text" warning | The PDF is likely scanned/image-based; use a text-based PDF |
| ChromaDB errors | `pip install --upgrade chromadb` |
| Slow first run | Normal — the embedding model downloads once (~90MB) |

## 🤝 Contributing

Issues and pull requests are welcome. For larger changes, please open an issue first to discuss what you'd like to change.

## 📄 License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and distribute, including commercially, with attribution.

## 👤 Author

**Awais Ur Rehman**
HIT Master's Student — Software Engineering
GitHub: https://github.com/AwaisRehman/AI-Research-Paper-Agent.git
Research focus: agentic software engineering tools

If you use this project in academic work, please cite it — see `CITATION.cff`.

## 🙏 Acknowledgements

- [Groq](https://groq.com) — free LLaMA 3.3 70B inference API
- [HuggingFace](https://huggingface.co) — `all-MiniLM-L6-v2` sentence embeddings
- [ChromaDB](https://www.trychroma.com) — open-source vector database
- [Streamlit](https://streamlit.io) — web app framework
