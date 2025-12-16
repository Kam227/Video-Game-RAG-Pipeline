# 🎮 Video Game RAG Assistant

## Project Description

This project implements a Retrieval-Augmented Generation (RAG) assistant for the video game domain. The system answers open-ended questions about video games by combining a large language model with semantic retrieval over a curated collection of video game reviews, factual descriptions, and community discussions.

Rather than relying solely on the model’s internal knowledge, the assistant grounds its responses in retrieved documents and explicitly displays the sources and similarity scores used to generate each answer. This improves accuracy, transparency, and user trust compared to standard chatbot systems.

---

## Domain Overview & Problem Statement

### Domain: Video Games

The video game domain contains a large volume of unstructured, opinionated, and rapidly changing information. Much of what players care about—game quality, balance, enjoyment, and long-term value—comes from reviews and forum discussions rather than authoritative sources.

### Problem

General-purpose language models often:
- Hallucinate details about games
- Provide shallow or generic recommendations
- Fail to reflect real player sentiment

Users have no way to verify whether an answer is grounded in actual data.

### Solution

This project addresses these issues by using a retrieval-augmented generation pipeline that:
- Retrieves semantically relevant passages from a vector database
- Grounds responses in real player-written content
- Displays retrieved sources with similarity scores for transparency

---

## Architecture Overview

The system follows a Retrieval-Augmented Generation (RAG) pipeline with agent-controlled retrieval.

### Pipeline

```
User Question
      ↓
LLM Agent
      ↓ (optional tool call)
Vector Database (DuckDB)
      ↓
Retrieved Passages + Similarity Scores
      ↓
Final Answer + Sources
```

### Key Components

- **LLM Agent**: Decides when retrieval is necessary and generates the final response.
- **Vector Database (DuckDB)**: Stores embedded documents and performs cosine similarity search.
- **Retrieval Tool**: Allows the agent to query the database when additional context is needed.
- **Streamlit UI**: Displays answers, sources, and similarity scores to the user.

---

## Document Collection Summary

The document collection consists of:
- Player reviews
- Community forum discussions
- Factual descriptions of video games

These documents were chosen because they reflect real player experiences, opinions, and recurring themes that are often missing from official marketing material.

All documents are embedded using a sentence-transformer model and stored as vectors in DuckDB, enabling semantic search rather than keyword matching.

---

## Agent Configuration & Rationale

The agent is explicitly customized for the video game domain.

### Agent Configuration

- **Role**: Video Game Content Assistant  
- **Goal**: Answer questions about video games using a database of reviews, facts, and forum discussions  
- **Backstory**: An expert assistant with access to curated video game data  

### Rationale

- Domain-specific prompting improves relevance and reduces generic responses.
- Allowing the agent to decide when retrieval is needed prevents unnecessary database queries.
- Forcing a final answer after tool usage ensures consistent user experience.
- Retrieved passages and similarity scores are preserved and displayed to improve transparency and trust.

---

## Installation & Setup Instructions

### Prerequisites

- Python 3.10+
- DuckDB
- OpenAI API key

### Installation

Clone the repository:

```bash
git clone https://github.com/your-username/video-game-rag-assistant.git
cd video-game-rag-assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up your vector database:
- Ensure your DuckDB file (e.g. `vg_vector.duckdb`) is placed in the correct directory
- Update `DEFAULT_DB_PATH` in `config.py` if necessary

### Running the App Locally

```bash
streamlit run app.py
```

Enter your OpenAI API key in the sidebar when prompted.

---

## Streamlit Deployment

**Live App**  
[https://your-streamlit-app-url.streamlit.app](https://vg-rag.streamlit.app/)

---

## Notes

- Retrieved sources are displayed with similarity scores to indicate relevance.
- Cosine distance is converted to similarity for interpretability.
- The system is designed to be lightweight, transparent, and easily extensible.

---

## Future Improvements

- Expand the document collection
- Add metadata filtering (genre, platform, release year)
- Improve ranking and evaluation of retrieved passages
- Add streaming responses and performance metrics
