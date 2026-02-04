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
    # Switched to 8b-instant for higher rate limits and stability
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.1-8b-instant", 
        groq_api_key=api_key,
        max_tokens=1000
    )
    return embeddings, vector_db, llm

embeddings, vector_db, llm = load_assets()

# 2. UI: CUSTOM CSS (FORCED 16.3px FONT & GOLD TITLE)
st.markdown(f"""
    <style>
    /* Global Font: Locked at 16.3px */
    html, body, [class*="st-"], .stApp p, .stApp span, .stApp div, .stApp li {{
        font-size: 16.3px !important;
        line-height: 1.5 !important;
        color: #000000 !important;
    }}

    /* Main Background: White */
    .stApp {{
        background-color: #FFFFFF !important;
    }}

    /* The Golden Title: Specific ID Selector */
    h1#locarno-pardo-d-oro-expert, .main-title {{
        color: #FFD700 !important;
        -webkit-text-fill-color: #FFD700 !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important; 
        margin-bottom: 1rem !important;
    }}

    /* Sidebar: Grey background with dark text */
    [data-testid="stSidebar"] {{
        background-color: #E6E9EF !important;
        min-width: 24rem !important;
        max-width: 24rem !important;
    }}
    [data-testid="stSidebar"] * {{
        font-size: 15.5px !important;
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

# 4. PROMPT DESIGN (Murakami Style + Defensive Logic)
template = """
You are a professional expert on the Locarno Film Festival. 
Context: {context}
Question: {question}

Instructions:
1. ONLY answer if the context explicitly contains the information requested.
2. For Year-specific questions: If the context does not contain the EXACT year mentioned (e.g., asked about 3022 but context only has 2022), state that you have no record for that year.
3. For Personal/General questions: If the question is an opinion or not related to festival records, politely state: "I am an AI assistant specialized in Locarno Film Festival history, and I don't have personal opinions or data beyond the records."
4. Your style should be professional, friendly, and reminiscent of Haruki Murakami's tone.
5. Identify ALL matching films. For EACH film, use this format:
   - **Movie Title & Winning Year**: [Title] ([Year])
   - **Director**: [Name]
   - **Country**: [Country Name]
   - **Summary**: [2-sentence summary]
   - **Source**: [Wikipedia URL]
6. IMPORTANT: If a summary mentions disambiguation (e.g., "may refer to 1919"), IGNORE those historical references and only report the movie matching the winning year.
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# 5. SIDEBAR: Portfolio Documentation
with st.sidebar:
    st.markdown("## Project Portfolio")
    
    st.markdown("### Technical Stack")
    st.markdown("""
    - **LLM:** Llama-3.1-8b-instant (Optimized for speed & rate limits)
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
st.markdown('<h1 class="main-title" id="locarno-pardo-d-oro-expert">🎬 Locarno Pardo d\'Oro Expert</h1>', unsafe_allow_html=True)
st.write("Specialized AI assistant for the official history of Golden Leopard winners.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_input := st.chat_input("Ask about winners..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # SEARCH LOGIC
    year_match = re.search(r"(\d{4})", user_input)
    is_complex = any(w in user_input.lower() for w in ["after", "since", "japanese", "china", "list", "all", "history"])
    
    final_docs = []
    all_docs = list(vector_db.docstore._dict.values())

    if year_match:
        target_year = year_match.group(1)
        final_docs = [d for d in all_docs if d.metadata.get("year") == target_year]

    if not final_docs:
        docs_with_score = vector_db.similarity_search_with_relevance_scores(user_input, k=15)
        # Similarity threshold to prevent irrelevant hallucinations
        final_docs = [doc for doc, score in docs_with_score if score > 0.40]

    context_text = "\n\n".join([
        f"TITLE: {d.metadata['title']}\nYEAR: {d.metadata['year']}\nDIRECTOR: {d.metadata['director']}\nURL: {d.metadata['url']}\nSUMMARY: {d.page_content}"
        for d in final_docs
    ])

    # AI INFERENCE
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Archives..."):
            full_prompt = QA_CHAIN_PROMPT.format(context=context_text, question=user_input)
            response = llm.invoke(full_prompt)
            answer = response.content
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
