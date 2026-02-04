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
# 4. PROMPT DESIGN (Format Enforcement Version)
template = """
You are a professional expert on the Locarno Film Festival history.
Context:
{context}

Question: {question}

Instructions:
1. ONLY answer based on the provided Context. 
2. If multiple movies match, you MUST list EVERY one of them individually.
3. If no matching movies exist in the Context, strictly state you have no record.
4. DO NOT use a conversational summary. Use the EXACT format below for EACH film:

FORMAT PER MOVIE:
- **Movie Title & Winning Year**: [Title] ([Year])
- **Director**: [Name]
- **Country**: [Country Name]
- **Summary**: [2-sentence summary]
- **Source**: [Wikipedia URL]

---
Begin listing the matching films now:
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

# 7. SEARCH & INFERENCE LOGIC (Supports Single Year, Range "After", and Range "Between")
if user_input := st.chat_input("Ask about winners..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # A. Advanced Extraction: Find all 4-digit years
    years = re.findall(r"(\d{4})", user_input)
    is_range_query = any(w in user_input.lower() for w in ["after", "since", "post", "recent", "history", "between", "from", "to", "-"])
    
    all_docs = list(vector_db.docstore._dict.values())
    final_docs = []
    logic_triggered = False

    # B. Enhanced Logic Layer
    if years:
        logic_triggered = True
        # Convert found years to integers
        year_ints = [int(y) for y in years]
        
        for d in all_docs:
            doc_year = int(d.metadata.get("year", 0))
            
            # Condition 1: Range "Between" (e.g., 1990 and 2000)
            if len(year_ints) >= 2:
                start_y, end_y = min(year_ints), max(year_ints)
                if start_y <= doc_year <= end_y:
                    final_docs.append(d)
            
            # Condition 2: Range "After" (e.g., after 1995)
            elif is_range_query:
                if doc_year > year_ints[0]:
                    final_docs.append(d)
            
            # Condition 3: Exact Year (e.g., in 2015)
            else:
                if doc_year == year_ints[0]:
                    final_docs.append(d)

    # C. Semantic Fallback
    if not final_docs and not logic_triggered:
        final_docs = vector_db.similarity_search(user_input, k=15)

    # D. Keyword Refinement (General for Korean, Japanese, etc.)
    keywords_to_check = ["korean", "japanese", "chinese", "french", "swiss", "italian"]
    active_keyword = next((k for k in keywords_to_check if k in user_input.lower()), None)
    
    if active_keyword and final_docs:
        search_term = "japan" if active_keyword == "japanese" else active_keyword.replace("ese", "").replace("ean", "")
        final_docs = [
            d for d in final_docs 
            if search_term in str(d.metadata.get("country", "")).lower() 
            or search_term in d.page_content.lower()
        ]

    # E. Context construction (Safe Cap)
    if len(final_docs) > 15:
        final_docs = final_docs[:15]

    if final_docs:
        context_text = "\n\n".join([
            f"TITLE: {d.metadata.get('title')}\nYEAR: {d.metadata.get('year')}\nDIRECTOR: {d.metadata.get('director')}\nCOUNTRY: {d.metadata.get('country')}\nURL: {d.metadata.get('url')}\nSUMMARY: {d.page_content}"
            for d in final_docs
        ])
    else:
        context_text = ""

    # F. Inference
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Archives..."):
            try:
                full_prompt = QA_CHAIN_PROMPT.format(context=context_text, question=user_input)
                response = llm.invoke(full_prompt)
                answer = response.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("API error or context too large. Please try a narrower search.")
