# 🎬 Locarno Pardo d'Oro Expert

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
- **Hybrid Retrieval Strategy:** Combines **Regex-based hard filtering** for specific year lookups with **Semantic Search** for thematic and descriptive queries (optimized with `k=15` for high recall).
- **Self-Explaining Metadata:** Enhanced vector chunks with explicit entity labels (Director, Country, Year) to improve retrieval accuracy for complex queries.
- **Cost-Effective Design:** Implemented a **Zero-Token Chat History** using Streamlit's local session state, maintaining a seamless UI experience without incurring extra API costs.
- **Data Integrity:** Fully automated ingestion pipeline from Wikipedia Open Data with forced source citations to eliminate AI hallucinations.

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
