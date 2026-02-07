# 🎬 Chatbot for Movie Search

An intelligent RAG (Retrieval-Augmented Generation) assistant specializing in the history of the **Locarno Film Festival's Golden Leopard** winners.

## Live Demo
**[Experience the AI Expert here!](https://locarno-movie-expert.streamlit.app/)**

## Technical Architecture
- **LLM:** [Llama-3.3-70B](https://groq.com) via Groq Cloud (State-of-the-art reasoning)
- **Vector Database:** [FAISS](https://github.com) (High-performance similarity search)
- **Embeddings:** `all-MiniLM-L6-v2` (Sentence Transformers)
- **Orchestration:** LangChain
- **Frontend:** Streamlit

## Key Engineering Features
- **Hybrid Retrieval Strategy with Execution Logic Feedback:** Combines **Regex-based hard filtering** for precise year/range lookups with **Semantic Search** for thematic queries. Integrated a **real-time execution status** (st.status) to provide transparency on which retrieval logic was triggered.
- **Multi-Stage Precision Filtering:** Implemented an automated "Guardrail" layer that cross-references user queries with metadata (Director, Country, Year) to eliminate hallucinations. Includes a **strict cross-country filter** (e.g., ensuring "Korean" queries only return South Korean cinema).
- **Source-Grounded Transparency:** Enhanced user trust by implementing an **Evidence Expander (st.expander)** that displays raw vector chunks and Wikipedia source URLs before final inference, ensuring every answer is fully traceable.
- **Production-Ready Containerization:** Fully Dockerized environment using a lightweight, CPU-optimized Python build. Includes a secure injection mechanism for API keys via environment variables, ensuring zero-risk deployment on platforms like AWS or local servers.
- **Zero-Cost State Management:** Utilizes Streamlit's local session state for chat history and result caching, maintaining a seamless UI experience without incurring extra token costs for history storage.
- **Data Integrity Pipeline:** Leverages Wikipedia Open Data with an automated ingestion pipeline, strictly enforcing source citations to eliminate AI hallucinations and ensure factual accuracy.

## Data Compliance
- Data is ethically sourced from Wikipedia's Golden Leopard archives via official APIs. All responses provide direct source links for verification.

## Docker Deployment
- **PREREQUISITES:** 
A valid GROQ API Key.
Create a file named '.env' in the root directory and add:
GROQ_API_KEY=your_actual_groq_api_key_here

Ensure your directory contains:
- app.py (Main logic)
- Dockerfile (Build instructions)
- .dockerignore (Excludes local junk)
- requirements.txt (Python dependencies)
- .env (Your private API Key)
- faiss_index/ (Vector database folder)

- **Run the Docker:** 
*Note: This may take a few minutes as it installs Torch and other dependencies.*
   ```bash
   docker build -t locarno-bot-app .
   docker run -p 8501:8501 --env-file .env locarno-bot-app
URL: http://localhost:8501

## Developed by yuan-phd
