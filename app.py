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
    
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.1-8b-instant",
        groq_api_key=api_key
    )
    return embeddings, vector_db, llm

embeddings, vector_db, llm = load_assets()

# 2. UI: CUSTOM CSS
st.markdown(f"""
    <style>
    html, body, [class*="st-"] {{
        font-size: 101.875% !important;
    }}
    .stApp {{
        background-color: #FFFFFF !important;
    }}
    .stApp p, .stApp div, .stApp span, .stApp li {{
        color: #000000 !important;
    }}
    .main-title {{
        color: #FFD700 !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        margin-bottom: 1rem !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: #E6E9EF !important;
        min-width: 24rem !important;
        max-width: 24rem !important;
    }}
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
You are a professional expert on the Locarno Film Festival history.
Context:
{context}

Question: {question}

Instructions:
1. ONLY answer based on the provided Context.
2. If the Context is empty or does not contain any movie matching the requested year or criteria, strictly state that you have no record for that request.
3. Do NOT use outside knowledge to invent movies or details.
4. For matching films, use this format:
   - **Movie Title & Winning Year**: [Title] ([Year])
   - **Director**: [Name]
   - **Country**: [Country Name]
   - **Summary**: [Summary]
   - **Source**: [URL]
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# 5. SIDEBAR (Fully Restored)
with st.sidebar:
    st.markdown("## Project Portfolio")
    st.markdown("### Technical Stack")
    st.markdown("""
    - **LLM:** Llama-3.1-8B (Groq)
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
    - **Efficiency:** Client-side session state for history
    """)
    st.divider()
    st.markdown("### Developer Contact")
    st.markdown("GitHub: [yuan-phd](https://github.com)")
    st.caption(f"Index Status: {vector_db.index.ntotal} Verified Records")

# 6. MAIN CONTENT
st.markdown('<h1 class="main-title">🎬 Locarno Pardo d\'Oro Expert</h1>', unsafe_allow_html=True)
st.write("Specialized AI assistant for the official history of Golden Leopard winners.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. SEARCH & INFERENCE LOGIC (Fixed Logic Order)
if user_input := st.chat_input("Ask about winners..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # A. Extract intent and years
    year_match = re.search(r"(\d{4})", user_input)
    is_range_query = any(w in user_input.lower() for w in ["after", "since", "post", "recent", "history"])
    
    all_docs = list(vector_db.docstore._dict.values())
    final_docs = []
    logic_triggered = False

    # B. Logic Layer: Hard Filtering for Years
    if year_match:
        target_year = int(year_match.group(1))
        logic_triggered = True
        if is_range_query:
            final_docs = [d for d in all_docs if int(d.metadata.get("year", 0)) > target_year]
        else:
            final_docs = [d for d in all_docs if int(d.metadata.get("year", 0)) == target_year]

    # C. Semantic Layer Fallback
    if not final_docs and not logic_triggered:
        final_docs = vector_db.similarity_search(user_input, k=15)

    # D. Keyword Refinement (REVISED: General keyword matching for Korean, Japanese, etc.)
    # This prevents sending non-relevant movies (like China/France) to the LLM
    keywords_to_check = ["korean", "japanese", "chinese", "french", "swiss", "italian"]
    active_keyword = next((k for k in keywords_to_check if k in user_input.lower()), None)
    
    if active_keyword and final_docs:
        # Map keywords to metadata short-codes if necessary (e.g., 'korean' -> 'korea')
        search_term = "japan" if active_keyword == "japanese" else active_keyword.replace("ese", "").replace("ean", "")
        final_docs = [
            d for d in final_docs 
            if search_term in str(d.metadata.get("country", "")).lower() 
            or search_term in d.page_content.lower()
        ]

    # E. Context construction (Limiting to top 15 to prevent API errors)
    if len(final_docs) > 15:
        final_docs = final_docs[:15]

    if final_docs:
        context_text = "\n\n".join([
            f"TITLE: {d.metadata.get('title')}\nYEAR: {d.metadata.get('year')}\nDIRECTOR: {d.metadata.get('director')}\nCOUNTRY: {d.metadata.get('country')}\nURL: {d.metadata.get('url')}\nSUMMARY: {d.page_content}"
            for d in final_docs
        ])
    else:
        context_text = ""

    # F. Inference with Error Handling
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Archives..."):
            try:
                full_prompt = QA_CHAIN_PROMPT.format(context=context_text, question=user_input)
                response = llm.invoke(full_prompt)
                answer = response.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("The search context is too large or the API is busy. Try a more specific year.")

