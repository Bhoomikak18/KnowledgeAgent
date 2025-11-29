# 📘 Knowledge Agent (RAG-Based Document Question Answering)

This project is a **Knowledge Base AI Agent** built using RAG (Retrieval Augmented Generation).  
It can read any PDF document, extract information, answer questions, and generate a knowledge graph of relationships.

This demo uses a *Ramayana PDF* only as a **sample dataset**.  
The same architecture can work for **HR policies, training content, SQL/Python notes, product manuals**, or any business documents.

---

##  Features
- Upload and process PDF documents
- Automatic text extraction using `pdfplumber`
- Chunking + vector embeddings using `bge-base-en`
- Semantic search using **ChromaDB**
- LLM reasoning using **Groq (Llama 3.1 8B Instant)**
- RAG-based accurate answers
- Relationship extraction using regex
- Auto-generated **knowledge graph** (PyVis + NetworkX)
- Simple **Gradio UI** for interacting with the agent

---

##  Architecture Overview

![Architecture Diagram](https://github.com/Bhoomikak18/KnowledgeAgent/blob/main/KnowledgeBaseAgent.png)

1. **PDF Extraction**  
   - Reads all PDFs inside `./pdfs/`
   - Extracts raw text using `pdfplumber`

2. **Text Processing & Chunking**  
   - Cleans text  
   - Splits into chunks using `RecursiveCharacterTextSplitter`

3. **Embeddings + Vector Store**  
   - Generates dense embeddings using `BAAI/bge-base-en-v1.5`  
   - Stores vectors inside **ChromaDB**

4. **RAG Query Pipeline**  
   - User enters a question  
   - System retrieves top-k matching chunks  
   - Injects them into a prompt  
   - Sends request to **Groq LLM**

5. **Answer Generation**  
   - LLM responds using context-based reasoning  
   - Output is factual and grounded in the PDF

6. **Triplet Extraction & Knowledge Graph**  
   - Regex extracts relationships (e.g., *Sita → wife → Rama*)  
   - Graph rendered using PyVis + NetworkX  
   - Displayed in Gradio

7. **Gradio UI**  
   - Chat interface  
   - Knowledge graph viewer

---

## 🛠 Tech Stack

### **AI / ML**
- Groq API (Llama 3.1 8B Instant)
- BGE Embeddings
- ChromaDB Vector Store

### **Python Libraries**
- `langchain`
- `pdfplumber`
- `gradio`
- `networkx`
- `pyvis`
- `re`
- `logging`

### **UI**
- Gradio Blocks (Web Interface)

---

# 📁 Project Structure
```
KnowledgeAgent/
│── agent.py                 # Main application file
│── README.md                # Project documentation
│── .gitignore               # Ignored files for Git
│── output_graph.html        # Generated graph preview (optional)
│── pdfs/                    # Source documents for RAG
│   └── ramayana.pdf
│
├── logs/                    # Runtime logs (ignored in Git)
├── chroma_db/               # Vector database files (ignored in Git)
└── venv/                    # Python virtual environment (ignored in Git)
```

---

# ⚙️ How to Run Locally

### **1. Create a Virtual Environment**
```bash
python -m venv venv
```

### **2. Activate the Virtual Environment**

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```bash
venv\Scripts\activate.bat
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Add Your Groq API Key**
**PowerShell:**
```bash
setx GROQ_API_KEY "YOUR_KEY_HERE"
```

### **5. Run the Agent**
```bash
python agent.py
```

### **6. Open the Gradio Link**
After running, you'll see:

```
Running on local URL: http://127.0.0.1:7860
Running on public URL: https://xxxx.gradio.live
```

Open any link in your browser.

---

# 🔄 How to Replace the PDF (Important)

If you want to use this agent for **HR policies, Python learning, SQL notes, onboarding guides, or support knowledge base**:

### **1. Delete the old PDF**
```
pdfs/ramayana.pdf
```

### **2. Add your document**
```
pdfs/my_document.pdf
```

### **3. Run the agent again**
```bash
python agent.py
```

It will automatically rebuild the **Chroma Vector DB**.

---

# ⚠️ Limitations

- Works only with text-based PDFs  
- Scanned PDFs are not supported  
- Regex relationship extraction is basic  
- Large PDFs take time to embed  
- Internet required for Groq LLM  

---

# 🚀 Future Improvements

- Support multiple PDFs  
- PDF upload directly from UI  
- Database-backed chat history  
- Authentication system  
- Deployment on HuggingFace / Render  
- Text-to-speech and speech-to-text  
- Advanced NLP relationship extraction  
- Domain expansion: HR / Python / SQL / Cloud / Networking
