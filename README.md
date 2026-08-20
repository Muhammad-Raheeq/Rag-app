#  RAG Document Assistant (Powered by Gemini API)

A Streamlit-based Retrieval-Augmented Generation (RAG) application that allows users to upload documents (`.pdf`, `.docx`, `.txt`) and ask context-aware questions. The app utilizes **HuggingFace** embeddings locally for fast, quota-free vectorization and Google's **Gemini API** for response generation.

---

##  Features

* **Multi-Format Support:** Parse and index PDF, Word (`.docx`), and Plain Text (`.txt`) files.
* **Local Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace for fast, local vector storage without hitting API embedding limits.
* **Vector Search:** Powered by `ChromaDB` for rapid similarity search across text chunks.
* **Context-Aware RAG Chain:** Built with LangChain to handle chat history, question reformulation, and precise answer generation.
* **Secure API Configuration:** Ingests user Gemini API keys securely via the Streamlit UI with automatic whitespace sanitization.

---

##  Tech Stack

* **Frontend:** Streamlit
* **Framework:** LangChain (`langchain-google-genai`, `langchain-community`, `langchain-huggingface`)
* **LLM Engine:** Google Gemini API (`gemini-3.6-flash`)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Vector Store:** ChromaDB

---

##  Project Structure

```text
├── app.py              # Main Streamlit application script
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation