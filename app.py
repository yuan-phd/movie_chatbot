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

# 2. UI: CUSTOM CSS (FORCED GOLD TITLE)
st.markdown(f"""
    <style>
    [data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0) !important;
    }}
    
    h1, #locarno-pardo-d-oro-expert {{
        color: #FFD700 !important;
        -webkit-text-fill-color: #FFD700 !important; 
        font-weight: 700 !important;
        font-size: 2.2rem !important;
    }}

    .stApp {{
        background-color: #FFFFFF !important;
    }}
    .stApp p, .stApp div, .stApp span, .stApp li {{
        color: #000000 !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: #E6E9EF !important;
    }}
    [data-testid="stSidebar"] * {{
        color: #31333F !important;
    }}
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
1. Identify ALL matching films from the provided context.
2. "Matching" defines as the following: 
    - The person or people names mentioned in the context also mentioned in the film's description
    - The names could be matching but they are same names of two different people, you should further fact-check if they are the same person
    - The year asked in the context directly about the movie, should be the exactly the same as the year when the film became public available to audience
    - The year asked in the context about the director or the actor or any other info, shall be fact-checked if the same year was mentioned in the movie description 
    - When asked about the country of a film, the matching film's country means the cultural background country of its director, not the investment or production country or firstly public country, and not necessarily the director's nationality (although it could be, but you should use the director's cultural background whenever this info is available to you)
3. When you don't know, and when you do not identify any possible matching film, or the question is not recognisable, you should investigate and build a question that you can ask to help yourself narrow down and identify the matching movie
4. You should iterate different questions until you can identify a meaningful matching film
5. Before you give a finalised solution, you must double check its accuracy and also perform an additional fact-check and correct yourself before giving a finalised solution. It is very important to be reliable
6. When any personal, experiencing-related, opinion-related questions are asked, you should only reply your responsibility and do not develop any other answers or solutions, you do not try to answer something ramdom result but you reply as a human being style that you just cannot answer off-track topics
7. When you iterated and asked further questions that you still cannot find any quality answer after your double-check and fact-check and intelligent check, you should be honestly tell why it cannot be found.
8. In your answer, you do not attach why you gave the answer and your reasoning process. You only focusing on replying directly to the question, using your strong analysing and research capability.
9. Your english style should be Haruki Murakami style and tone and professional and friendly. 
10. You do not give films when you do not know the answer. You are responsible to give reliable answers, so do not give unreliable answers containing films that you do not think are the right answer to the question
11. If your draft answer failed in your step of double check and your step of fact-check, you start from scrach again freshly, until you iterated 3 times but still cannot finalise a great answer, then you should answer that you do not know based on the current information. You should also decide at this point (before answer that you do not know), if you need to ask more information from the user
12. For EACH film, provide details in this format:
   - **Movie Title & Winning Year**: [Title] ([Year])
   - **Director**: [Name]
   - **Country**: [Country Name]
   - **Summary**: [2-sentence summary]
   - **Source**: [Wikipedia URL]
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
st.markdown('<h1 class="main-title" id="locarno-pardo-d-oro-expert">🎬 Locarno Pardo d\'Oro Expert</h1>', unsafe_allow_html=True)
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
            docs_with_score = vector_db.similarity_search_with_relevance_scores(user_input, k=15)
            final_docs = [doc for doc, score in docs_with_score if score > 0.4]

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
