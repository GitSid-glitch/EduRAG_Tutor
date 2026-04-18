# RAG Tutor

RAG Tutor is a FastAPI-based internship assignment submission for grounded PDF question answering with inline topic images. A user uploads a chapter PDF, the app builds a local retrieval index, and every answer is generated only from the top retrieved document chunks plus an optional relevant diagram.

## Features

- Upload any chapter PDF and create a topic index locally
- Ask grounded questions tied only to the uploaded document
- Retrieve top `K=3-5` relevant chunks before answer generation
- Use Groq for final answer generation with strict document-only prompting
- Show a relevant image inline for the curated Sound chapter demo
- Display supporting passages in a compact expandable section

## Tech stack

- Backend: FastAPI
- Frontend: vanilla HTML, CSS, JavaScript
- PDF extraction: `pypdf`
- Retrieval: local embeddings with `sentence-transformers` when available, hashing fallback otherwise
- LLM: Groq Chat Completions API

## API endpoints

- `POST /upload`
  - Uploads a PDF, extracts text, chunks it, creates embeddings, and returns `topicId`
- `POST /chat`
  - Accepts `topicId`, `message`, and optional `history`, retrieves relevant chunks, and returns a grounded answer
- `GET /images/{topicId}`
  - Returns image metadata linked to the indexed topic

## RAG pipeline explanation

1. The user uploads a PDF chapter.
2. The backend extracts page text with `pypdf`.
3. Text is normalized, cleaned, and split into smaller sentence-window chunks.
4. Each chunk gets a local embedding vector.
5. For each question, the app retrieves the most relevant chunks using hybrid ranking:
   - semantic similarity
   - lexical overlap
   - phrase matching
   - sentence-level relevance
   - noise/question-fragment penalties
6. Only the retrieved chunks are sent to Groq.
7. The model returns a short grounded answer, and the UI shows supporting passages separately.

## Image retrieval logic

The assignment specifically asks for a small JSON-style diagram library for the topic. This implementation follows that approach for the provided NCERT Sound chapter:

- a curated set of Sound diagrams is stored as local metadata plus assets
- each image has `title`, `keywords`, and `description`
- the backend compares the user question plus top retrieved text against this metadata
- the best matching image is returned inline with the answer

For other uploaded PDFs, text RAG still works. If no topic image set is available, the app simply returns no image instead of forcing an irrelevant one.

## Prompts used

System prompt:

> You are a grounded AI tutor. Answer only from the retrieved document context. Give a short, clean answer in 2-4 sentences, not a transcript or chunk dump. If the answer is partially missing, clearly say what is supported and what is not. Use friendly teaching language and cite pages inline. Never use outside knowledge.

User prompt structure:

> Document title, user question, selected image candidate, and retrieved context only.

## Local run steps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set this in `.env`:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

Run:

```bash
uvicorn app.main:app --reload --port 8001
```

Then open `http://127.0.0.1:8001`.

## Deployment

This project is ready to deploy as a normal FastAPI web service. Recommended target: Render.

### Render deployment steps

1. Push this repo to GitHub.
2. Create a new Web Service on Render.
3. Connect the GitHub repo.
4. Use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables:
   - `GROQ_API_KEY`
   - `GROQ_MODEL`

## Demo suggestions

- Upload the provided NCERT Sound chapter
- Ask:
  - `What is vibration?`
  - `Why are sound waves called mechanical waves?`
  - `What is pitch?`
  - `What is echo?`
- Show the answer, image, and supporting passages

## Deliverables alignment

- GitHub repo: this project
- Working chatbot: local now, deployable to Render
- README: includes RAG pipeline, image retrieval logic, prompts, and run steps
- Demo video: recommended 2-4 minute walkthrough of upload, chat, and image retrieval
