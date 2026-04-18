from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pypdf import PdfReader
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:  
    SentenceTransformer = None


app = FastAPI(title="RAG Tutor")


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "how",
    "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "what", "when", "where", "which", "who", "why", "with", "you", "your",
}

OCR_MERGE_PROTECT = STOPWORDS | {
    "we", "he", "she", "they", "them", "our", "ours", "his", "her", "hers",
    "their", "theirs", "then", "do", "does", "did", "done", "set", "next",
    "make", "list", "hear", "any", "many", "more", "most", "some", "each",
    "have", "has", "had", "will", "would", "should", "could", "been", "being",
    "after", "before", "above", "below", "into", "unto", "upon", "about",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")
    return cleaned[:48] or "document"


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}", text.lower()) if t not in STOPWORDS]


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _clip_text(text: str, limit: int = 220) -> str:
    compact = _normalize(text)
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _clip_clean_sentence(text: str, limit: int = 220) -> str:
    compact = _normalize(text)
    if len(compact) <= limit:
        return compact

    sentences = _split_sentences(compact)
    if not sentences:
        return _clip_text(compact, limit)

    chosen: list[str] = []
    total = 0
    for sentence in sentences:
        projected = total + len(sentence) + (1 if chosen else 0)
        if chosen and projected > limit:
            break
        if not chosen and len(sentence) > limit:
            trimmed = sentence[:limit].rsplit(" ", 1)[0].rstrip(",;:-")
            return trimmed + "..."
        chosen.append(sentence)
        total = projected

    if chosen:
        return " ".join(chosen).rstrip(",;:-")
    return _clip_text(compact, limit)


def _query_ngrams(tokens: list[str]) -> list[str]:
    grams: list[str] = []
    for size in (3, 2):
        for index in range(len(tokens) - size + 1):
            grams.append(" ".join(tokens[index : index + size]))
    return grams


def _is_noise_line(line: str) -> bool:
    lowered = line.lower().strip()
    if not lowered:
        return False
    if re.fullmatch(r"(sound|science)\s*\d+[a-z0-9 ]*", lowered):
        return True
    if re.fullmatch(r"reprint\s+\d{4}[-\u2013]\d{2}", lowered):
        return True
    if re.fullmatch(r"\d+", lowered):
        return True
    return False


def _repair_ocr_splits(line: str) -> str:
    pattern = re.compile(r"\b([A-Za-z]{1,8})\s+([A-Za-z]{1,16})\b")
    repaired = line

    while True:
        changed = False

        def replacer(match: re.Match[str]) -> str:
            nonlocal changed
            left, right = match.group(1), match.group(2)
            if left.lower() in OCR_MERGE_PROTECT or right.lower() in OCR_MERGE_PROTECT:
                return match.group(0)
            if len(left) <= 2 or len(right) <= 2:
                changed = True
                return left + right
            return match.group(0)

        updated = pattern.sub(replacer, repaired)
        if not changed:
            updated = re.sub(r"\bfr omotion\b", "fro motion", updated, flags=re.IGNORECASE)
            return updated
        repaired = updated


def _normalize_page_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = [_repair_ocr_splits(re.sub(r"\s+", " ", line).strip()) for line in text.split("\n")]
    paragraphs: list[str] = []
    current: list[str] = []

    for line in raw_lines:
        if _is_noise_line(line):
            continue
        if not line:
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue
        current.append(line)

    if current:
        paragraphs.append(" ".join(current).strip())

    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def _definition_target(query: str) -> str:
    lowered = query.strip().lower().rstrip(" ?.")
    patterns = (
        r"^what is (?:an? |the )?(?P<term>.+)$",
        r"^what are (?:the )?(?P<term>.+)$",
        r"^define (?P<term>.+)$",
        r"^meaning of (?P<term>.+)$",
        r"^explain (?P<term>.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, lowered)
        if match:
            return _normalize(match.group("term"))
    return ""


def _query_intent(query: str) -> str:
    q = query.lower().strip()
    if re.search(r"\b(define|what is|what are|meaning of)\b", q):
        return "definition"
    if re.search(r"\b(why|how|explain)\b", q):
        return "explanation"
    if re.search(r"\b(calculate|find|compute|how many|value|time period|wavelength|frequency)\b", q):
        return "numerical"
    if re.search(r"\b(compare|difference|distinguish|between)\b", q):
        return "comparison"
    if re.search(r"\b(used for|application|applications|used in)\b", q):
        return "application"
    return "general"


def _load_json_if_exists(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))

@dataclass
class EmbeddingBackend:
    base_dir: Path
    model_name: str = "all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        self.vectorizer = HashingVectorizer(n_features=768, alternate_sign=False, norm="l2")
        self.model = None
        try:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers unavailable")
            cache_dir = self.base_dir / ".model_cache"
            cache_dir.mkdir(exist_ok=True)
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=str(cache_dir),
                local_files_only=True,
            )
            self.backend_name = f"sentence-transformers:{self.model_name}"
        except Exception:
            self.backend_name = "hashing-fallback"

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.model is not None:
            return np.asarray(self.model.encode(texts, normalize_embeddings=True))
        return self.vectorizer.transform(texts).toarray()


# -----------------------------
# Tutor engine
# -----------------------------

class TutorEngine:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.storage_dir = base_dir / "storage"
        self.storage_dir.mkdir(exist_ok=True)
        self.upload_dir = self.storage_dir / "uploads"
        self.upload_dir.mkdir(exist_ok=True)
        self.topics_dir = self.storage_dir / "topics"
        self.topics_dir.mkdir(exist_ok=True)
        self.embedding_backend = EmbeddingBackend(base_dir=base_dir)
        self.sound_images = self._load_sound_images()

    async def ingest_pdf(self, file: UploadFile, images_metadata: Optional[list[dict]] = None) -> dict:
        topic_id = uuid4().hex[:10]
        filename = file.filename or "document.pdf"
        safe_name = f"{topic_id}-{_slugify(filename)}.pdf"
        pdf_path = self.upload_dir / safe_name
        content = await file.read()
        pdf_path.write_bytes(content)

        reader = PdfReader(str(pdf_path))
        pages: list[dict[str, Any]] = []
        for idx, page in enumerate(reader.pages, start=1):
            text = _normalize_page_text(page.extract_text() or "")
            if text:
                pages.append({"page": idx, "text": text})

        if not pages:
            raise HTTPException(status_code=400, detail="No extractable text found in the uploaded PDF.")

        chunks = self._build_chunks(pages)
        chunk_vectors = self.embedding_backend.encode([chunk["text"] for chunk in chunks])
        for chunk, vector in zip(chunks, chunk_vectors):
            chunk["embedding"] = vector.tolist()

        topic_dir = self.topics_dir / topic_id
        topic_dir.mkdir(exist_ok=True)

        topic_meta = {
            "topicId": topic_id,
            "title": Path(filename).stem,
            "filename": filename,
            "pages": len(pages),
            "chunks": len(chunks),
            "embeddingBackend": self.embedding_backend.backend_name,
            "pdfPath": str(pdf_path),
            "digest": hashlib.sha256(content).hexdigest()[:16],
            "summary": self._summarize_document(chunks),
            "imagesAvailable": images_metadata or self._topic_images(filename, chunks),
        }

        (topic_dir / "topic.json").write_text(json.dumps(topic_meta, indent=2), encoding="utf-8")
        (topic_dir / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
        if images_metadata is not None:
            (topic_dir / "images.json").write_text(json.dumps(images_metadata, indent=2), encoding="utf-8")

        return topic_meta

    async def answer_question(self, topic_id: str, message: str, history: list[dict[str, str]]) -> dict:
        topic = self.get_topic(topic_id)
        chunks = self._load_chunks(topic_id)
        retrieved = self._retrieve(chunks=chunks, query=message, limit=5)
        image = self._select_image(topic=topic, question=message, chunks=retrieved)
        answer = self._generate_answer(topic=topic, question=message, retrieved_chunks=retrieved, history=history, image=image)

        return {
            "answer": self._finalize_answer(answer),
            "image": image,
            "sources": [
                {
                    "chunkId": chunk["chunkId"],
                    "page": chunk["page"],
                    "score": round(chunk["score"], 4),
                    "excerpt": self._source_excerpt(chunk["text"], message),
                }
                for chunk in retrieved[:3]
            ],
            "grounding": {
                "retrievedCount": len(retrieved),
                "embeddingBackend": topic["embeddingBackend"],
            },
        }

    def get_topic(self, topic_id: str) -> dict:
        topic_file = self.topics_dir / topic_id / "topic.json"
        if not topic_file.exists():
            raise HTTPException(status_code=404, detail="Unknown topicId")

        topic = json.loads(topic_file.read_text(encoding="utf-8"))
        images_file = self.topics_dir / topic_id / "images.json"
        if images_file.exists():
            topic["imagesAvailable"] = json.loads(images_file.read_text(encoding="utf-8"))
        return topic

    def get_images(self, topic_id: str) -> dict:
        topic = self.get_topic(topic_id)
        return {"topicId": topic_id, "images": topic.get("imagesAvailable", [])}

    def _load_chunks(self, topic_id: str) -> list[dict]:
        path = self.topics_dir / topic_id / "chunks.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Chunks not found for topicId")
        return json.loads(path.read_text(encoding="utf-8"))
    def _build_chunks(self, pages: list[dict[str, Any]]) -> list[dict]:
        chunks: list[dict] = []
        chunk_id = 1
        for page in pages:
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", page["text"]) if p.strip()]
            page_chunks = self._chunk_paragraphs(paragraphs)
            for text in page_chunks:
                chunks.append(
                    {
                        "chunkId": f"chunk_{chunk_id:03d}",
                        "page": page["page"],
                        "text": text,
                    }
                )
                chunk_id += 1
        return chunks

    def _chunk_paragraphs(self, paragraphs: list[str]) -> list[str]:
        chunks: list[str] = []
        for paragraph in paragraphs:
            sentences = _split_sentences(paragraph)
            if not sentences:
                continue

            if len(paragraph) <= 420 and len(sentences) <= 4:
                chunks.append(paragraph)
                continue

            start = 0
            while start < len(sentences):
                window: list[str] = []
                char_count = 0
                index = start
                while index < len(sentences):
                    sentence = sentences[index]
                    projected = char_count + len(sentence) + (1 if window else 0)
                    if window and (projected > 420 or len(window) >= 3):
                        break
                    window.append(sentence)
                    char_count = projected
                    index += 1

                chunk_text = " ".join(window).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                if index >= len(sentences):
                    break
                start = max(start + 1, index - 1)

        return chunks

    def _retrieve(self, chunks: list[dict], query: str, limit: int) -> list[dict]:
        intent = _query_intent(query)
        query_vec = self.embedding_backend.encode([query])[0]
        chunk_vectors = np.asarray([chunk["embedding"] for chunk in chunks])
        semantic = cosine_similarity([query_vec], chunk_vectors)[0]

        query_tokens = _tokenize(query)
        query_counts = Counter(query_tokens)
        query_unique = set(query_tokens)
        query_phrases = _query_ngrams(query_tokens)
        definition_target = _definition_target(query)

        scored: list[dict] = []
        for chunk, sem_score in zip(chunks, semantic):
            chunk_score = self._score_chunk(
                chunk=chunk,
                semantic_score=float(sem_score),
                query_counts=query_counts,
                query_unique=query_unique,
                query_phrases=query_phrases,
                definition_target=definition_target,
                intent=intent,
            )
            scored.append({**chunk, "score": chunk_score})

        scored.sort(key=lambda item: item["score"], reverse=True)
        return self._select_diverse_chunks(scored, limit)

    def _score_chunk(
        self,
        chunk: dict,
        semantic_score: float,
        query_counts: Counter[str],
        query_unique: set[str],
        query_phrases: list[str],
        definition_target: str,
        intent: str,
    ) -> float:
        chunk_tokens = _tokenize(chunk["text"])
        chunk_counts = Counter(chunk_tokens)
        chunk_lower = chunk["text"].lower()

        coverage = 0.0
        if query_unique:
            coverage = len([token for token in query_unique if token in chunk_counts]) / len(query_unique)

        lexical = 0.0
        if query_counts:
            lexical = sum(min(query_counts[token], chunk_counts.get(token, 0)) for token in query_counts) / max(1, len(query_counts))

        phrase_hits = 0.0
        if query_phrases:
            phrase_hits = sum(1 for phrase in query_phrases if phrase in chunk_lower) / len(query_phrases)

        sentence_scores = [
            self._sentence_relevance(
                sentence,
                query_counts,
                query_unique,
                query_phrases,
                definition_target,
                intent,
            )
            for sentence in _split_sentences(chunk["text"])
        ]
        best_sentence = max(sentence_scores, default=0.0)

        definition_bonus = self._definition_bonus(chunk["text"], definition_target)
        intent_bonus = self._intent_bonus(chunk["text"], intent)
        question_penalty = self._question_like_penalty(chunk["text"])
        noise_penalty = self._noise_penalty(chunk["text"])

        semantic_weight = 0.50 if coverage < 0.35 else 0.38
        score = (
            semantic_weight * semantic_score
            + 0.18 * coverage
            + 0.14 * lexical
            + 0.10 * phrase_hits
            + 0.13 * best_sentence
            + 0.08 * definition_bonus
            + 0.05 * intent_bonus
        )
        return float(max(0.0, score - question_penalty - noise_penalty))

    def _sentence_relevance(
        self,
        sentence: str,
        query_counts: Counter[str],
        query_unique: set[str],
        query_phrases: list[str],
        definition_target: str,
        intent: str,
    ) -> float:
        sentence_tokens = _tokenize(sentence)
        if not sentence_tokens:
            return 0.0

        sentence_counts = Counter(sentence_tokens)
        sentence_lower = sentence.lower()

        overlap = 0.0
        if query_unique:
            overlap = len([token for token in query_unique if token in sentence_counts]) / len(query_unique)

        lexical = 0.0
        if query_counts:
            lexical = sum(min(query_counts[token], sentence_counts.get(token, 0)) for token in query_counts) / max(1, len(query_counts))

        phrase_hits = 0.0
        if query_phrases:
            phrase_hits = sum(1 for phrase in query_phrases if phrase in sentence_lower) / len(query_phrases)

        intent_bonus = self._intent_bonus(sentence, intent)
        definition_bonus = self._definition_bonus(sentence, definition_target)
        score = 0.50 * overlap + 0.30 * lexical + 0.12 * phrase_hits + intent_bonus + definition_bonus
        score -= self._question_like_penalty(sentence)
        score -= self._noise_penalty(sentence)
        return float(max(0.0, score))

    def _intent_bonus(self, text: str, intent: str) -> float:
        lowered = text.lower()
        if intent == "definition":
            if any(p in lowered for p in [" is ", " are ", " is called ", " are called ", " means ", " refers to ", " defined as "]):
                return 0.08
            return 0.0
        if intent == "explanation":
            if any(p in lowered for p in [" because ", " due to ", " as a result ", " therefore ", " hence ", " this happens "]):
                return 0.06
        if intent == "numerical":
            if any(p in lowered for p in [" = ", " formula ", " speed ", " wavelength ", " frequency ", " time period "]):
                return 0.06
        if intent == "application":
            if any(p in lowered for p in [" used for ", " used to ", " application ", " employed ", " helps "]):
                return 0.06
        if intent == "comparison":
            if any(p in lowered for p in [" differs ", " compared to ", " while ", " whereas ", " both "]):
                return 0.05
        return 0.0

    def _definition_bonus(self, text: str, definition_target: str) -> float:
        if not definition_target:
            return 0.0
        lowered = text.lower()
        target = re.escape(definition_target)
        patterns = [
            rf"\b{target}\b\s+(is|are|means|refers to|denotes)\b",
            rf"\b(is|are|called|known as|defined as)\b.{0,35}\b{target}\b",
        ]
        if any(re.search(pattern, lowered) for pattern in patterns):
            return 0.12
        return 0.0

    def _question_like_penalty(self, text: str) -> float:
        lowered = text.lower()
        penalty = 0.0
        penalty += min(text.count("?"), 3) * 0.08
        if re.search(r"\bquestions?\b|\bexercises?\b", lowered):
            penalty += 0.18
        if re.search(r"(^|\s)(what|how|why|when|which|explain|suppose|will|can|does|do)\b", lowered):
            penalty += 0.06
        return min(penalty, 0.34)

    def _noise_penalty(self, text: str) -> float:
        compact = text.strip()
        lowered = compact.lower()
        penalty = 0.0
        if compact.startswith("•"):
            penalty += 0.04
        if re.match(r"^(fig|table|\d+\.\d+[:.]?)", lowered):
            penalty += 0.12
        if len(compact) < 35:
            penalty += 0.12
        return min(penalty, 0.18)

    def _select_diverse_chunks(self, candidates: list[dict], limit: int) -> list[dict]:
        selected: list[dict] = []
        pool = candidates[: max(limit * 4, limit)]

        while pool and len(selected) < limit:
            if not selected:
                selected.append(pool.pop(0))
                continue

            best_index = 0
            best_value = float("-inf")
            for index, candidate in enumerate(pool):
                similarity = max(
                    cosine_similarity([candidate["embedding"]], [chosen["embedding"]])[0][0]
                    for chosen in selected
                )
                mmr_score = 0.82 * candidate["score"] - 0.18 * float(similarity)
                if mmr_score > best_value:
                    best_value = mmr_score
                    best_index = index

            selected.append(pool.pop(best_index))

        return selected

    def _retrieval_confidence(self, retrieved_chunks: list[dict], question: str) -> float:
        if not retrieved_chunks:
            return 0.0
        top = retrieved_chunks[0]["score"]
        avg = sum(chunk["score"] for chunk in retrieved_chunks[:3]) / min(3, len(retrieved_chunks))
        query_tokens = set(_tokenize(question))
        all_tokens = set(_tokenize(" ".join(chunk["text"] for chunk in retrieved_chunks[:3])))
        coverage = len(query_tokens & all_tokens) / max(1, len(query_tokens))
        return 0.55 * top + 0.25 * avg + 0.20 * coverage
    def _generate_answer(
        self,
        topic: dict,
        question: str,
        retrieved_chunks: list[dict],
        history: list[dict[str, str]],
        image: dict | None,
    ) -> str:
        confidence = self._retrieval_confidence(retrieved_chunks, question)
        if confidence < 0.22:
            return "I could not find enough grounded material in the uploaded document to answer this confidently."

        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                return self._groq_answer(topic, question, retrieved_chunks, history, image, groq_key)
            except Exception:
                pass

        return self._grounded_fallback(topic, question, retrieved_chunks, image)

    def _groq_answer(
        self,
        topic: dict,
        question: str,
        retrieved_chunks: list[dict],
        history: list[dict[str, str]],
        image: dict | None,
        groq_key: str,
    ) -> str:
        prompt_context = "\n\n".join(
            f"[Page {chunk['page']} | score {chunk['score']:.3f}] {chunk['text']}" for chunk in retrieved_chunks[:4]
        )
        short_history = history[-4:]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a grounded AI tutor. Answer only from the retrieved document context. "
                    "Do not use outside knowledge, even if you know it. "
                    "If the document does not clearly support the answer, say that the uploaded chapter does not clearly answer it. "
                    "Keep the answer short, clean, and factual. Use friendly teaching language. "
                    "Cite page numbers inline like (p. 3)."
                ),
            },
            *short_history,
            {
                "role": "user",
                "content": (
                    f"Document title: {topic['title']}\n"
                    f"Question: {question}\n"
                    f"Relevant image candidate: {image['title'] if image else 'None'}\n\n"
                    "Rules:\n"
                    "- Use only the context below.\n"
                    "- Do not add concepts that are not in the document.\n"
                    "- If the answer is not fully supported, say so.\n"
                    "- Prefer 2 to 4 sentences.\n\n"
                    f"Retrieved context:\n{prompt_context}"
                ),
            },
        ]
        payload = json.dumps(
            {
                "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                "temperature": 0.1,
                "messages": messages,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {groq_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()

    def _grounded_fallback(self, topic: dict, question: str, retrieved_chunks: list[dict], image: dict | None) -> str:
        intent = _query_intent(question)
        query_tokens = _tokenize(question)
        query_counts = Counter(query_tokens)
        query_unique = set(query_tokens)
        query_phrases = _query_ngrams(query_tokens)
        definition_target = _definition_target(question)

        candidates: list[tuple[float, str, int]] = []
        for chunk in retrieved_chunks[:3]:
            for sentence in _split_sentences(chunk["text"]):
                if self._question_like_penalty(sentence) >= 0.12:
                    continue
                rel = self._sentence_relevance(
                    sentence,
                    query_counts,
                    query_unique,
                    query_phrases,
                    definition_target,
                    intent,
                )
                candidates.append((rel, sentence, chunk["page"]))

        ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
        selected_sentences: list[str] = []

        if intent == "definition":
            for _, sentence, _ in ranked:
                if len(sentence) > 15:
                    selected_sentences.append(_clip_text(sentence, 190))
                    break
        else:
            for _, sentence, _ in ranked[:2]:
                if sentence.strip():
                    selected_sentences.append(_clip_text(sentence, 190))

        if not selected_sentences:
            answer = "I could not find enough grounded material in the uploaded document to answer confidently."
        else:
            answer = " ".join(selected_sentences)

        page_refs = ", ".join(sorted({f"p. {chunk['page']}" for chunk in retrieved_chunks[:2]}))
        image_hint = f" The visual helps illustrate this idea: {image['title']}." if image else ""
        return f"{answer} ({page_refs}){image_hint}"

    def _finalize_answer(self, answer: str) -> str:
        cleaned = _normalize(answer)
        if len(cleaned) <= 320:
            return cleaned
        sentences = _split_sentences(cleaned)
        if len(sentences) >= 2:
            return _clip_clean_sentence(" ".join(sentences[:2]), 320)
        return _clip_clean_sentence(cleaned, 320)

    def _source_excerpt(self, text: str, question: str) -> str:
        sentences = _split_sentences(text)
        if not sentences:
            return _clip_clean_sentence(text, 160)
        intent = _query_intent(question)
        query_tokens = _tokenize(question)
        query_counts = Counter(query_tokens)
        query_terms = set(query_tokens)
        query_phrases = _query_ngrams(query_tokens)
        definition_target = _definition_target(question)
        ranked = sorted(
            sentences,
            key=lambda sentence: self._sentence_relevance(
                sentence,
                query_counts,
                query_terms,
                query_phrases,
                definition_target,
                intent,
            ),
            reverse=True,
        )
        best = ranked[0] if ranked else sentences[0]
        return _clip_clean_sentence(best, 160)

    def _summarize_document(self, chunks: list[dict]) -> str:
        preview = " ".join(chunk["text"] for chunk in chunks[:3])
        sentences = _split_sentences(preview)
        return " ".join(sentences[:3])[:360].strip()

    def _topic_images(self, filename: str, chunks: list[dict]) -> list[dict]:
        combined = f"{filename.lower()} " + " ".join(chunk["text"] for chunk in chunks[:24]).lower()
        sound_terms = [
            "sound",
            "vibration",
            "amplitude",
            "frequency",
            "pitch",
            "echo",
            "compression",
            "rarefaction",
            "tuning fork",
            "longitudinal",
            "loudness",
            "eardrum",
        ]
        hits = sum(1 for term in sound_terms if term in combined)
        if "sound" in filename.lower() or hits >= 2:
            return self.sound_images
        return []

    def _load_sound_images(self) -> list[dict]:
        return [
            {
                "id": "img_001",
                "filename": "/static/assets/bell-vibration.svg",
                "title": "Bell Vibration",
                "keywords": ["bell", "vibration", "sound", "source"],
                "description": "A bell producing sound through rapid vibration.",
            },
            {
                "id": "img_002",
                "filename": "/static/assets/tuning-fork.svg",
                "title": "Tuning Fork Oscillation",
                "keywords": ["tuning fork", "oscillation", "prongs", "vibration"],
                "description": "A tuning fork shows alternating oscillation that creates sound waves.",
            },
            {
                "id": "img_003",
                "filename": "/static/assets/longitudinal-wave.svg",
                "title": "Longitudinal Sound Wave",
                "keywords": ["wave", "compression", "rarefaction", "medium"],
                "description": "Sound traveling as longitudinal compressions and rarefactions in air.",
            },
            {
                "id": "img_004",
                "filename": "/static/assets/ear-pathway.svg",
                "title": "How The Ear Detects Sound",
                "keywords": ["ear", "eardrum", "hearing", "sound"],
                "description": "The outer ear, eardrum, and inner ear pathway used for hearing.",
            },
            {
                "id": "img_005",
                "filename": "/static/assets/amplitude-pitch.svg",
                "title": "Amplitude And Pitch",
                "keywords": ["amplitude", "pitch", "loudness", "frequency"],
                "description": "Comparison of wave amplitude and frequency to explain loudness and pitch.",
            },
            {
                "id": "img_006",
                "filename": "/static/assets/echo.svg",
                "title": "Echo Reflection",
                "keywords": ["echo", "reflection", "distance", "time"],
                "description": "Sound reflection from a wall leading to an echo heard by the listener.",
            },
        ]

    def _select_image(self, topic: dict, question: str, chunks: list[dict]) -> dict | None:
        images = topic.get("imagesAvailable", []) or []
        if not images:
            return None

        query = question + " " + " ".join(chunk["text"][:220] for chunk in chunks[:2])
        query_vec = self.embedding_backend.encode([query])[0]
        image_texts = [
            f"{img.get('title', '')} {' '.join(img.get('keywords', []))} {img.get('description', '')}"
            for img in images
        ]
        image_vecs = self.embedding_backend.encode(image_texts)
        sims = cosine_similarity([query_vec], image_vecs)[0]

        question_tokens = set(_tokenize(question + " " + " ".join(chunk["text"][:120] for chunk in chunks[:2])))
        combined_scores: list[float] = []
        for sim, image in zip(sims, images):
            keywords = [k.lower() for k in image.get("keywords", [])]
            keyword_hits = len([keyword for keyword in keywords if keyword in question.lower()])
            token_hits = len([token for token in question_tokens if token in " ".join(keywords) or token in image.get("title", "").lower()])
            lexical_bonus = min(0.24, 0.08 * keyword_hits + 0.03 * token_hits)
            combined_scores.append(float(sim) + lexical_bonus)

        best_idx = int(np.argmax(combined_scores))
        if combined_scores[best_idx] < 0.22:
            return None
        return images[best_idx]


engine = TutorEngine(base_dir=Path(__file__).resolve().parent)


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    images_json: str = Form(""),
):
    images_metadata = None
    if images_json.strip():
        try:
            parsed = json.loads(images_json)
            if not isinstance(parsed, list):
                raise ValueError
            images_metadata = parsed
        except Exception:
            raise HTTPException(status_code=400, detail="images_json must be a JSON array of image metadata objects.")

    topic = await engine.ingest_pdf(file=file, images_metadata=images_metadata)
    return {"topicId": topic["topicId"], "title": topic["title"], "pages": topic["pages"], "chunks": topic["chunks"]}


@app.post("/chat")
async def chat(payload: dict):
    topic_id = payload.get("topicId")
    message = payload.get("message", "")
    history = payload.get("history", []) or []

    if not topic_id or not message:
        raise HTTPException(status_code=400, detail="topicId and message are required.")
    if not isinstance(history, list):
        raise HTTPException(status_code=400, detail="history must be a list.")

    result = await engine.answer_question(topic_id=topic_id, message=message, history=history)
    return result


@app.get("/images/{topic_id}")
def get_images(topic_id: str):
    return engine.get_images(topic_id)


@app.get("/topic/{topic_id}")
def get_topic(topic_id: str):
    return engine.get_topic(topic_id)
