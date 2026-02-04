import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# 1. SETUP & CONFIGURATION
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

@st.cache_resource
def load_assets():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
    return embeddings, vector_db, llm

embeddings, vector_db, llm = load_assets()

# 2. UI: CUSTOM CSS
st.markdown(f"""
    <style>
    /* Global font: Precise 16.3px */
    html, body, [class*="st-"] {{
        font-size: 101.875% !important;
    }}

    /* MAIN AREA: White background with Black text */
    .stApp {{
        background-color: #FFFFFF !important;
    }}
    .stApp p, .stApp div, .stApp span, .stApp li {{
        color: #000000 !important;
    }}

    /* THE GOLDEN TITLE: Cinematic Gold Header */
    .main-title {{
        color: #FFD700 !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        margin-bottom: 1rem !important;
    }}

    /* SIDEBAR: Widen by 15% (24rem) and Grey background */
    [data-testid="stSidebar"] {{
        background-color: #E6E9EF !important;
        min-width: 24rem !important;
        max-width: 24rem !important;
    }}
    
    /* SIDEBAR TEXT: Clear contrast for technical details */
    [data-testid="stSidebar"] * {{
        color: #31333F !important;
    }}
    [data-testid="stSidebar"] a {{
        color: #31333F !important;
        text-decoration: underline !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# 3. CHAT HISTORY LOGIC
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. PROMPT DESIGN
template = """
You are a professional expert on the Locarno Film Festival. 

Context:
{context}

Question: {question}

Instructions:
1. ONLY answer if the context explicitly contains the information requested.
2. For Question 1 (Year specific): If the context does not contain the EXACT year mentioned in the question (e.g., if asked about 3022 but context only has 2022), you MUST state that you have no record for that year. Do NOT substitute with other years.
3. For Question 2 (Personal/General): If the question is a personal opinion or a general question not related to the specific festival records in the context, politely state: "I am an AI assistant specialized in Locarno Film Festival history, and I don't have personal opinions or data beyond the festival records."
4. If a match is found, use the standard bold format with Wikipedia source.

Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# 5. SIDEBAR: Detailed Portfolio Layout
with st.sidebar:
    st.markdown("## Project Portfolio")
    
    st.markdown("### Technical Stack")
    st.markdown("""
    - **LLM:** Llama-3.3-70B (Groq)
    - **Vector Store:** FAISS (Facebook AI Similarity Search)
    - **Embeddings:** all-MiniLM-L6-v2 (Sentence Transformers)
    - **Logic:** Hybrid Retrieval Strategy (Regex + Semantic)
    """)

    st.markdown("### Data Compliance")
    st.markdown("""
    - **Source:** Wikipedia Open Data
    - **Acquisition:** Official Wikipedia API
    - **Attribution:** Source-linked citations provided
    """)

    st.markdown("### Deployment")
    st.markdown("""
    - **Platform:** Streamlit Community Cloud
    - **Security:** Secrets Management for API keys
    - **Efficiency:** Zero-Token UI Chat History
    """)
    
    st.divider()
    
    st.markdown("### Developer Contact")
    st.markdown("GitHub: [yuan-phd](https://github.com)")
    st.caption(f"Index Status: {vector_db.index.ntotal} Verified Records")

# 6. MAIN CONTENT
st.markdown('<h1 class="main-title">🎬 Locarno Pardo d\'Oro Expert</h1>', unsafe_allow_html=True)
st.write("Specialized AI assistant for the official history of Golden Leopard winners.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_input := st.chat_input("Ask about winners (e.g. 'Who won in 2000?')"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # SEARCH LOGIC
        # --- SEARCH LOGIC (REPLACEMENT) ---
    year_match = re.search(r"(\d{4})", user_input)
    is_complex = any(w in user_input.lower() for w in ["after", "since", "japanese", "china", "list", "all", "history"])
    
    final_docs = []
    
    if year_match:
        target_year = year_match.group(1)
        all_docs = vector_db.docstore._dict.values()
        exact_year_docs = [d for d in all_docs if d.metadata.get("year") == target_year]
        
        if exact_year_docs:
            final_docs = exact_year_docs
        elif not is_complex:
            final_docs = []
            
    if not final_docs:
        if not (year_match and not is_complex):
            final_docs = vector_db.similarity_search(user_input, k=15)

    context_text = "\n\n".join([
        f"TITLE: {d.metadata['title']}\nYEAR: {d.metadata['year']}\nDIRECTOR: {d.metadata['director']}\nURL: {d.metadata['url']}\nSUMMARY: {d.page_content}"
        for d in final_docs
    ])

    with st.chat_message("assistant"):
        with st.spinner("Analyzing Archives..."):
            full_prompt = QA_CHAIN_PROMPT.format(context=context_text, question=user_input)
            response = llm.invoke(full_prompt)
            answer = response.content
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
