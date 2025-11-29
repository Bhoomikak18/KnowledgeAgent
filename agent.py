# ============================================================
# KNOWLEDGE AGENT - FINAL PERFECT WORKING VERSION (2025)
# ============================================================

import os
import re
import pdfplumber
import gradio as gr
import networkx as nx
from pyvis.network import Network
import logging

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from groq import Groq


# ============================================================
# LOGGING
# ============================================================

os.makedirs("./logs", exist_ok=True)

logging.basicConfig(
    filename='./logs/agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

logging.info("=== Knowledge Agent (Groq Direct Version) Initialized ===")


# ============================================================
# PDF → TEXT EXTRACTION
# ============================================================

def load_pdf_text(folder="./pdfs"):
    logging.info("Extracting PDF text...")
    text_data = ""

    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            path = os.path.join(folder, filename)
            logging.info(f"Reading PDF: {path}")

            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_data += t + "\n"

    logging.info(f"PDF extraction complete. Total characters: {len(text_data)}")
    return text_data


raw_text = load_pdf_text()


# ============================================================
# CLEAN + CHUNK
# ============================================================

clean_text = re.sub(r'\s+', ' ', raw_text).strip()

splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
chunks = splitter.split_text(clean_text)

logging.info(f"Total chunks created: {len(chunks)}")


# ============================================================
# EMBEDDINGS + VECTOR DB
# ============================================================

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

vector_db = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

logging.info("Vector DB built successfully")


# ============================================================
# GROQ DIRECT API
# ============================================================

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def groq_llm(prompt_text):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0
    )
    return completion.choices[0].message.content


# ============================================================
# PROMPT TEMPLATE
# ============================================================

prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""
You are a Sanatan Dharma expert specializing in the Ramayana.

RULES:
1. Answer ONLY in THIRD PERSON.
2. When asked "who is X" or "what is X" or "about X", provide ALL available information including:
   - Family relationships (daughter of, son of, wife of, husband of)
   - Titles and roles (princess of, king of, prince of)
   - Important characteristics or affiliations
3. Use proper story-style factual sentences:
   - "Sita is the daughter of King Janaka."
   - "Sita is the wife of Rama."
   - "Sita is the princess of Mithila."
4. For relationships ALWAYS use these exact formats:
   - "X is the daughter of Y"
   - "X is the son of Y"
   - "X is the wife of Y"
   - "X is the husband of Y"
   - "X is the princess of Y"
   - "X is the prince of Y"
   - "X is the king of Y"
   - "X is the brother of Y"
5. List multiple facts in separate sentences.
6. Use ONLY the given context. If not found say: "Information not found."
7. Keep answer respectful & comprehensive.

Context:
{context}

Question: {question}

Chat History:
{chat_history}

Answer:
"""
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# ============================================================
# RAG PIPELINE
# ============================================================

def rag_answer(question, chat_history):

    # boost retrieval for identity questions and main characters
    is_identity_question = any(word in question.lower() for word in ["who is", "what is", "about", "tell me about"])
    has_main_character = any(name in question.lower() for name in ["sita", "rama", "lakshman", "hanuman", "ravana", "janaka", "dasharatha"])
    
    k_value = 12 if (is_identity_question or has_main_character) else 6

    docs = vector_db.similarity_search(question, k=k_value)
    context = "\n".join([d.page_content for d in docs])

    final_prompt = prompt.format(
        context=context,
        question=question,
        chat_history=chat_history
    )

    answer = groq_llm(final_prompt)

    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(answer)

    return answer


# ============================================================
# TRIPLET EXTRACTION (FINAL FIXED VERSION)
# ============================================================

def extract_triplets(text):
    triplets = []

    patterns = [
        r"([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*) is the ([a-zA-Z ]+) of ([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)",
        r"([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*) is daughter of ([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)",
        r"([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*) is son of ([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)",
        r"([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*) is wife of ([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)",
        r"([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*) married ([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)",
        
        # ⭐ MOST IMPORTANT FIX — detects "Sita of Videha"
        r"([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*) of ([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)"
    ]
    
    seen = set()  # avoid duplicates

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            if len(m) == 3:
                subj, rel, obj = m[0].strip(), m[1].strip(), m[2].strip()
                triplet = (subj, rel, obj)
                if triplet not in seen:
                    triplets.append(triplet)
                    seen.add(triplet)
            elif len(m) == 2:
                subj, obj = m[0].strip(), m[1].strip()
                triplet = (subj, "from", obj)
                if triplet not in seen:
                    triplets.append(triplet)
                    seen.add(triplet)

    logging.info(f"Extracted {len(triplets)} triplets: {triplets}")
    return triplets


# ============================================================
# KNOWLEDGE GRAPH RENDERING (FINAL FIXED)
# ============================================================

def build_graph_html(triplets):
    if not triplets:
        return "<h3 style='color:gray;text-align:center;padding:20px;'>No relationships found in the answer.</h3>"

    G = nx.DiGraph()

    for s, r, o in triplets:
        G.add_edge(s, o, label=r, title=f"{s} → {r} → {o}")

    # If graph has no edges, show message
    if len(G.edges()) == 0:
        return "<h3 style='color:gray;text-align:center;padding:20px;'>No relationships could be visualized.</h3>"

    net = Network(height="550px", width="100%", directed=True, bgcolor="#FFFFFF")

    net.set_options("""
    {
      "nodes": {
        "shape": "dot", 
        "size": 25, 
        "font": {"size": 16, "color": "#333333"},
        "color": {"background": "#97C2FC", "border": "#2B7CE9"},
        "borderWidth": 2
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}, 
        "font": {"size": 12, "align": "middle"},
        "color": {"color": "#848484"},
        "smooth": {"type": "continuous"}
      },
      "physics": {
        "enabled": true, 
        "barnesHut": {"springLength": 150, "springConstant": 0.04, "damping": 0.09}
      }
    }
    """)

    net.from_nx(G)

    # Generate HTML directly
    html = net.generate_html()
    
    # Return simple iframe-ready HTML
    return f"""
    <iframe srcdoc='{html.replace("'", "&apos;")}' 
            width="100%" 
            height="550px" 
            frameborder="0" 
            style="border:1px solid #ccc;">
    </iframe>
    """


# ============================================================
# GRADIO UI
# ============================================================

def chatbot_fn(question):
    if not question:
        return "Ask your about Doubts", ""

    # Get chat history from memory
    chat_history = memory.load_memory_variables({}).get("chat_history", "")
    
    answer = rag_answer(question, chat_history)
    triplets = extract_triplets(answer)
    graph_html = build_graph_html(triplets)

    return answer, graph_html


with gr.Blocks() as ui:
    gr.Markdown("# Knowledge Agent")
    gr.Markdown("### Ask questions about your doubts and explore knowledge relationships")
    
    with gr.Column():
        question_input = gr.Textbox(
            label="Ask about your doubts", 
            placeholder="Example: Who is Sita?",
            lines=2
        )
        submit_btn = gr.Button("Submit", variant="primary", size="lg")
        
        answer_output = gr.Textbox(
            label="Answer", 
            lines=8
        )
        
        graph_output = gr.HTML(
            label="Knowledge Graph",
            value="<div style='text-align:center;padding:50px;color:#666;'>Submit a question to see the knowledge graph</div>"
        )
    
    submit_btn.click(
        fn=chatbot_fn,
        inputs=question_input,
        outputs=[answer_output, graph_output]
    )
    
    question_input.submit(
        fn=chatbot_fn,
        inputs=question_input,
        outputs=[answer_output, graph_output]
    )

ui.launch(share=True)
