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
    # 回归到你喜欢的 70b 强力模型
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile", 
        groq_api_key=api_key
    )
    return embeddings, vector_db, llm

embeddings, vector_db, llm = load_assets()

# 2. UI: 暴力锁定 16.3px (不使用相对单位)
st.markdown("""
    <style>
    /* 强制锁定全局正文、列表、段落、侧边栏文字为 16.3px */
    html, body, [class*="st-"], .stApp p, .stApp span, .stApp div, .stApp li {
        font-size: 16.3px !important;
        line-height: 1.5 !important;
    }

    /* 侧边栏背景与宽度 */
    [data-testid="stSidebar"] {
        background-color: #E6E9EF !important;
        min-width: 24rem !important;
        max-width: 24rem !important;
    }
    
    /* 侧边栏内的链接与文字颜色 */
    [data-testid="stSidebar"] * {
        color: #31333F !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. CHAT HISTORY LOGIC
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. PROMPT DESIGN
template = """
You are a professional expert on the Locarno Film Festival. 
Context: {context}
Question: {question}

Instructions:
1. ONLY answer if the context explicitly contains the information requested.
2. For Year-specific questions: If the context does not contain the EXACT year mentioned, state that you have no record for that year.
3. For Personal/General questions: Politely state that you are an AI assistant specialized in Locarno history.
4. Identify ALL matching films. For EACH film, use this format:
   - **Movie Title & Winning Year**: [Title] ([Year])
   - **Director**: [Name]
   - **Country**: [Country Name]
   - **Summary**: [2-sentence summary]
   - **Source**: [Wikipedia URL]
5. Your tone should be professional and informative.
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# 5. SIDEBAR: 技术栈展示
with st.sidebar:
    st.markdown("## Project Portfolio")
    st.markdown("### Technical Stack")
    st.markdown("""
    - **LLM:** Llama-3.3-70B (High Precision)
    - **Vector Store:** FAISS
    - **Embeddings:** all-MiniLM-L6-v2
    - **Logic:** Hybrid Retrieval Strategy
    """)
    st.markdown("### Data Compliance")
    st.markdown("""
    - **Source:** Wikipedia Open Data
    - **Acquisition:** Official Wikipedia API
    """)
    st.markdown("### Deployment")
    st.markdown("""
    - **Platform:** Streamlit Cloud
    - **Efficiency:** Zero-Token UI History
    """)
    st.divider()
    st.markdown("### Developer Contact")
    st.markdown("GitHub: [yuan-phd](https://github.com)")
    st.caption(f"Index Status: {vector_db.index.ntotal} Verified Records")

# 6. MAIN CONTENT
st.title("🎬 Locarno Pardo d'Oro Expert")
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
        # 使用 K=15 保证召回率
        docs_with_score = vector_db.similarity_search_with_relevance_scores(user_input, k=15)
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
