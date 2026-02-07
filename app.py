import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# 1. SETUP & CONFIGURATION
import os
from dotenv import load_dotenv

# Try to load .env, but won't fail if the file is missing in Docker
load_dotenv() 

# Prioritize environment variables injected by Docker
api_key = os.getenv("GROQ_API_KEY")

# Safety Check: If no key is found, show a clear error in the UI
if not api_key:
    st.error("Missing GROQ_API_KEY. Please set it in your environment variables.")
    st.stop()


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
# 4. PROMPT DESIGN (Format Enforcement Version)
template = """
You are a professional expert on the Locarno Film Festival history.
Context:
{context}

Question: {question}

Instructions:
1. ONLY answer based on the provided Context. 
2. If the user asks for a specific person (e.g., "Hong Sang-soo"), ONLY include movies in your list where that EXACT person is the director. 
3. If multiple movies match, list them individually without duplicates.
4. DO NOT include "related" directors or "similar" countries. If the director's name in the context is "Hong XX" and the user asked for "Hong Sang-soo", EXCLUDE "Hong XX".
5. DO NOT use a conversational summary. Use the EXACT format below for EACH film:

FORMAT PER MOVIE:
- **Movie Title & Winning Year**: [Title] ([Year])
- **Director**: [Name]
- **Country**: [Country Name]
- **Summary**: [2-sentence summary]
- **Source**: [Wikipedia URL]

---
Begin listing the matching films now:
Output ONLY the list. Do not provide alternative formats or extra commentary.
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
    - **Deduplication:** Regex-based Title Normalization
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

# 7. SEARCH & INFERENCE LOGIC (Ultra-Precise Version)
if user_input := st.chat_input("Ask about winners..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- 1. st.status to show retrieval logic ---
    with st.status("Analyzing Search Logic...", expanded=True) as status:
        years = re.findall(r"(\d{4})", user_input)
        is_range_query = any(w in user_input.lower() for w in ["after", "since", "post", "history", "between", "-"])
        
        all_docs = list(vector_db.docstore._dict.values())
        final_docs = []
        logic_triggered = False

        # A. Year Logic (Priority)
        if years:
            st.write(f"📅 Triggered Year Logic: {years}")
            logic_triggered = True
            y_ints = [int(y) for y in years]
            for d in all_docs:
                try:
                    doc_year = int(str(d.metadata.get("year", 0)))
                    if len(y_ints) >= 2:
                        if min(y_ints) <= doc_year <= max(y_ints): final_docs.append(d)
                    elif is_range_query:
                        if doc_year > y_ints[0]: final_docs.append(d)
                    else:
                        if doc_year == y_ints[0]: final_docs.append(d)
                except: continue

        # B. Semantic Search (Fallback)
        if not final_docs and not logic_triggered:
            st.write("🧠 Triggered Semantic Search (Vector DB)")
            final_docs = vector_db.similarity_search(user_input, k=20)

        # C. ULTIMATE PRECISION FILTER
        query_lower = user_input.lower()
        # Extract only important words (names/countries) from query
        stop_words = {"movie", "winner", "director", "the", "after", "since", "between", "from", "show", "list"}
        query_parts = {p for p in query_lower.split() if p not in stop_words and len(p) > 2}
        
        refined_results = []
        seen_titles = set()

        st.write("🔍 Applying Precision Filters (Name/Country/Dedup)...")
        for d in final_docs:
            m = d.metadata
            # Normalize Title for strict de-duplication
            raw_title = str(m.get('title', ''))
            clean_title = re.sub(r'\W+', '', raw_title).lower() # Remove all non-alphanumeric
            
            if clean_title in seen_titles: continue
            
            director = str(m.get('director', '')).lower()
            country = str(m.get('country', '')).lower()
            
            keep = True
            # 1. Name Match Logic
            if query_parts and not logic_triggered:
                match_count = sum(1 for p in query_parts if p in director or p in clean_title)
                if match_count == 0:
                    keep = False
                
                # 2. Cross-Country Guard
                if "korean" in query_lower and "korea" not in country:
                    keep = False
                if "chinese" in query_lower and "china" not in country:
                    keep = False

            if keep:
                refined_results.append(d)
                seen_titles.add(clean_title)

        final_docs = refined_results[:15]
        status.update(label="Search Strategy Applied", state="complete", expanded=False)

    # D. Context Construction
    if final_docs:
        context_text = "\n\n".join([
            f"TITLE: {d.metadata.get('title')}\nYEAR: {d.metadata.get('year')}\nDIRECTOR: {d.metadata.get('director')}\nCOUNTRY: {d.metadata.get('country')}\nURL: {d.metadata.get('url')}\nSUMMARY: {d.page_content}"
            for d in final_docs
        ])
    else:
        context_text = ""

    # E. Inference
    with st.chat_message("assistant"):
        # --- 2. st.expander to show data source ---
        if final_docs:
            with st.expander("📚 Source Documents"):
                for doc in final_docs:
                    st.write(f"**{doc.metadata.get('title')}** ({doc.metadata.get('year')})")
                    st.caption(f"Director: {doc.metadata.get('director')} | Country: {doc.metadata.get('country')}")

        with st.spinner("Refining search..."):
            try:
                full_prompt = QA_CHAIN_PROMPT.format(context=context_text, question=user_input)
                response = llm.invoke(full_prompt)
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as e:
                st.error("API Limit reached. Please try again.")
