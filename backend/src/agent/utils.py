from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def resolve_urls(urls_to_resolve: List[Any], id: int) -> Dict[str, str]:
    """
    Create a map of the vertex ai search urls (very long) to a short url with a unique id for each url.
    Ensures each original URL gets a consistent shortened form while maintaining uniqueness.
    """
    prefix = f"https://vertexaisearch.cloud.google.com/id/"
    urls = [site.web.uri for site in urls_to_resolve]

    # Create a dictionary that maps each unique URL to its first occurrence index
    resolved_map = {}
    for idx, url in enumerate(urls):
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{id}-{idx}"

    return resolved_map


def insert_citation_markers(text, citations_list):
    """
    Inserts citation markers into a text string based on start and end indices.

    Args:
        text (str): The original text string.
        citations_list (list): A list of dictionaries, where each dictionary
                               contains 'start_index', 'end_index', and
                               'segment_string' (the marker to insert).
                               Indices are assumed to be for the original text.

    Returns:
        str: The text with citation markers inserted.
    """
    # Sort citations by end_index in descending order.
    # If end_index is the same, secondary sort by start_index descending.
    # This ensures that insertions at the end of the string don't affect
    # the indices of earlier parts of the string that still need to be processed.
    sorted_citations = sorted(
        citations_list, key=lambda c: (c["end_index"], c["start_index"]), reverse=True
    )

    modified_text = text
    for citation_info in sorted_citations:
        # These indices refer to positions in the *original* text,
        # but since we iterate from the end, they remain valid for insertion
        # relative to the parts of the string already processed.
        end_idx = citation_info["end_index"]
        marker_to_insert = ""
        for segment in citation_info["segments"]:
            marker_to_insert += f" [{segment['label']}]({segment['short_url']})"
        # Insert the citation marker at the original end_idx position
        modified_text = (
            modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]
        )

    return modified_text


# ---------- TF-IDF local search helpers ----------
import re
import math
from collections import Counter, defaultdict

# Minimal stopword set to improve signal
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "have",
    "has",
    "but",
    "not",
    "you",
}


def _tokenize(text: str):
    """Simple tokenizer: lowercase, extract word tokens, filter stopwords and short tokens."""
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]


# Helper functions for vector computation (TF-IDF and hashed embeddings)

def _compute_tfidf_vector(tf: Counter, idf: dict):
    """Compute and normalize a TF-IDF vector from term frequencies and idf map.

    Returns:
        (vec: dict, norm: float) where vec is normalized (L2 norm = 1) or empty dict if norm=0.
    """
    vec = {}
    for term, freq in tf.items():
        vec[term] = freq * idf.get(term, 1.0)
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm > 0:
        for term in vec:
            vec[term] /= norm
    return vec, norm


def _build_query_vector_tfidf(qtokens: List[str], index: dict):
    """Build normalized query TF-IDF vector (dict) given tokenized query and index."""
    qtf = Counter(qtokens)
    N = index["N"]
    idf = index["idf"]
    qvec = {}
    for term, freq in qtf.items():
        term_idf = idf.get(term, math.log((N + 1) / 1) + 1.0)
        qvec[term] = freq * term_idf
    qnorm = math.sqrt(sum(v * v for v in qvec.values()))
    if qnorm > 0:
        for term in qvec:
            qvec[term] /= qnorm
    return qvec


def _compute_hashed_vector(tf: Counter, idf: dict, dim: int):
    """Compute and normalize a hashed embedding vector from tf and idf."""
    vec = [0.0] * dim
    for term, freq in tf.items():
        w = freq * idf.get(term, 1.0)
        idx = _hash_token_to_index(term, dim)
        vec[idx] += w
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec, norm


def _build_hashed_query_vector(qtokens: List[str], index: dict, N: int, dim: int):
    """Build normalized hashed query vector from tokenized query and index."""
    qtf = Counter(qtokens)
    qvec = [0.0] * dim
    for term, freq in qtf.items():
        term_idf = index["idf"].get(term, math.log((N + 1) / 1) + 1.0)
        w = freq * term_idf
        idx = _hash_token_to_index(term, dim)
        qvec[idx] += w
    qnorm = math.sqrt(sum(v * v for v in qvec))
    if qnorm > 0:
        qvec = [v / qnorm for v in qvec]
    return qvec


def build_tfidf_index(dir_path: str):
    """Build a small TF-IDF index over .md files in dir_path.

    Returns:
        dict with keys: 'docs' (list), 'idf' (dict), 'N' (doc count)
        Each doc is dict: {'path','title','text','tf' (Counter), 'vec' (dict), 'norm' (float)}
    """
    docs = []
    df = defaultdict(int)

    for root, _, files in os.walk(dir_path):
        for fname in files:
            if not fname.lower().endswith(".md"):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    text = fh.read()
            except Exception:
                continue

            # title: first H1 or filename
            title = None
            for line in text.splitlines():
                if line.strip().startswith("# "):
                    title = line.strip().lstrip("# ").strip()
                    break
            if not title:
                title = fname

            tokens = _tokenize(text)
            if not tokens:
                continue
            tf = Counter(tokens)
            for term in set(tf.keys()):
                df[term] += 1

            docs.append({"path": path, "title": title, "text": text, "tf": tf})

    N = len(docs)
    idf = {}
    for term, doccount in df.items():
        idf[term] = math.log((N + 1) / (doccount + 1)) + 1.0

    # compute normalized tf-idf vectors
    for d in docs:
        vec, norm = _compute_tfidf_vector(d["tf"], idf)
        d["vec"] = vec
        d["norm"] = norm

    return {"docs": docs, "idf": idf, "N": N}


def search_dir_tfidf(dir_path: str, query: str, top_k: int = 5):
    """Search the directory using TF-IDF and return ranked results.

    Returns list of dicts with title, snippet, url, confidence (score 0..1)
    """
    index = build_tfidf_index(dir_path)
    docs = index["docs"]
    N = index["N"]
    if N == 0:
        return []

    qtokens = _tokenize(query)
    if not qtokens:
        return []

    qvec = _build_query_vector_tfidf(qtokens, index)

    # score docs by dot product
    scored = []
    for d in docs:
        score = 0.0
        for term, qval in qvec.items():
            score += qval * d["vec"].get(term, 0.0)
        if score > 0:
            # find snippet around best matching term
            snippet = None
            text_lower = d["text"].lower()
            best_idx = None
            for term in qtokens:
                idx = text_lower.find(term.lower())
                if idx != -1 and (best_idx is None or idx < best_idx):
                    best_idx = idx
            if best_idx is not None:
                start = max(0, best_idx - 200)
                end = min(len(d["text"]), best_idx + 200)
                snippet = d["text"][start:end].replace("\n", " ").strip()
            else:
                snippet = (d["text"][:400].replace("\n", " ").strip())

            scored.append({
                "title": d["title"],
                "snippet": snippet,
                "url": f"file://{os.path.abspath(d['path'])}",
                "confidence": float(score),
            })

    # sort by score desc and return top_k
    scored.sort(key=lambda x: x["confidence"], reverse=True)
    # Normalize confidences to max 1.0 if any
    if scored:
        max_score = max(s["confidence"] for s in scored)
        if max_score > 0:
            for s in scored:
                s["confidence"] = s["confidence"] / max_score

    return scored[:top_k]


# ---------- Local hashed-embedding semantic search ----------
import xxhash
from typing import Tuple

# Simple in-memory cache to avoid rebuilding indexes for the same directory
_SEM_INDEX_CACHE: dict = {}


def _hash_token_to_index(token: str, dim: int) -> int:
    return xxhash.xxh64(token).intdigest() % dim


def build_semantic_index_local(dir_path: str, dim: int = 8192):
    """Build a semantic index using hashed token embeddings for .md files.

    Each token is hashed into a fixed-size vector of length `dim`. Token weights
    use TF * IDF to improve discrimination. Returns a dict with docs and idf.
    """
    if dir_path in _SEM_INDEX_CACHE:
        return _SEM_INDEX_CACHE[dir_path]

    docs = []
    df = defaultdict(int)

    for root, _, files in os.walk(dir_path):
        for fname in files:
            if not fname.lower().endswith(".md"):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    text = fh.read()
            except Exception:
                continue

            # title: first H1 or filename
            title = None
            for line in text.splitlines():
                if line.strip().startswith("# "):
                    title = line.strip().lstrip("# ").strip()
                    break
            if not title:
                title = fname

            tokens = _tokenize(text)
            if not tokens:
                continue
            tf = Counter(tokens)
            for term in set(tf.keys()):
                df[term] += 1

            docs.append({"path": path, "title": title, "text": text, "tf": tf})

    N = len(docs)
    if N == 0:
        _SEM_INDEX_CACHE[dir_path] = {"docs": [], "idf": {}, "N": 0, "dim": dim}
        return _SEM_INDEX_CACHE[dir_path]

    idf = {}
    for term, doccount in df.items():
        idf[term] = math.log((N + 1) / (doccount + 1)) + 1.0

    # Build hashed embeddings per document
    for d in docs:
        vec, norm = _compute_hashed_vector(d["tf"], idf, dim)
        d["vec"] = vec
        d["norm"] = norm

    index = {"docs": docs, "idf": idf, "N": N, "dim": dim}
    _SEM_INDEX_CACHE[dir_path] = index
    return index


def search_dir_semantic_local(dir_path: str, query: str, top_k: int = 5, dim: int = 8192):
    """Search the directory using local hashed embeddings and cosine similarity.

    Falls back to TF-IDF (search_dir_tfidf) if no documents are found.
    """
    index = build_semantic_index_local(dir_path, dim=dim)
    docs = index["docs"]
    N = index["N"]
    if N == 0:
        return search_dir_tfidf(dir_path, query, top_k=top_k)

    qtokens = _tokenize(query)
    if not qtokens:
        return []

    qvec = _build_hashed_query_vector(qtokens, index, N, dim)

    scored = []
    for d in docs:
        # dot product
        score = 0.0
        for i, val in enumerate(qvec):
            if val != 0.0:
                score += val * d["vec"][i]
        if score > 0:
            # token-based snippet for readability
            text_lower = d["text"].lower()
            best_idx = None
            for term in qtokens:
                idx = text_lower.find(term.lower())
                if idx != -1 and (best_idx is None or idx < best_idx):
                    best_idx = idx
            if best_idx is not None:
                start = max(0, best_idx - 200)
                end = min(len(d["text"]), best_idx + 200)
                snippet = d["text"][start:end].replace("\n", " ").strip()
            else:
                snippet = d["text"][:400].replace("\n", " ").strip()

            scored.append({
                "title": d["title"],
                "snippet": snippet,
                "url": f"file://{os.path.abspath(d['path'])}",
                "confidence": float(score),
            })

    if not scored:
        return []

    scored.sort(key=lambda x: x["confidence"], reverse=True)
    max_score = max(s["confidence"] for s in scored)
    if max_score > 0:
        for s in scored:
            s["confidence"] = s["confidence"] / max_score

    return scored[:top_k]


def get_citations(response, resolved_urls_map):
    """
    Extracts and formats citation information from a Gemini model's response.

    This function processes the grounding metadata provided in the response to
    construct a list of citation objects. Each citation object includes the
    start and end indices of the text segment it refers to, and a string
    containing formatted markdown links to the supporting web chunks.

    Args:
        response: The response object from the Gemini model, expected to have
                  a structure including `candidates[0].grounding_metadata`.
                  It also relies on a `resolved_map` being available in its
                  scope to map chunk URIs to resolved URLs.

    Returns:
        list: A list of dictionaries, where each dictionary represents a citation
              and has the following keys:
              - "start_index" (int): The starting character index of the cited
                                     segment in the original text. Defaults to 0
                                     if not specified.
              - "end_index" (int): The character index immediately after the
                                   end of the cited segment (exclusive).
              - "segments" (list[str]): A list of individual markdown-formatted
                                        links for each grounding chunk.
              - "segment_string" (str): A concatenated string of all markdown-
                                        formatted links for the citation.
              Returns an empty list if no valid candidates or grounding supports
              are found, or if essential data is missing.
    """
    citations = []

    # Ensure response and necessary nested structures are present
    if not response or not response.candidates:
        return citations

    candidate = response.candidates[0]
    if (
        not hasattr(candidate, "grounding_metadata")
        or not candidate.grounding_metadata
        or not hasattr(candidate.grounding_metadata, "grounding_supports")
    ):
        return citations

    for support in candidate.grounding_metadata.grounding_supports:
        citation = {}

        # Ensure segment information is present
        if not hasattr(support, "segment") or support.segment is None:
            continue  # Skip this support if segment info is missing

        start_index = (
            support.segment.start_index
            if support.segment.start_index is not None
            else 0
        )

        # Ensure end_index is present to form a valid segment
        if support.segment.end_index is None:
            continue  # Skip if end_index is missing, as it's crucial

        # Add 1 to end_index to make it an exclusive end for slicing/range purposes
        # (assuming the API provides an inclusive end_index)
        citation["start_index"] = start_index
        citation["end_index"] = support.segment.end_index

        citation["segments"] = []
        if (
            hasattr(support, "grounding_chunk_indices")
            and support.grounding_chunk_indices
        ):
            for ind in support.grounding_chunk_indices:
                try:
                    chunk = candidate.grounding_metadata.grounding_chunks[ind]
                    resolved_url = resolved_urls_map.get(chunk.web.uri, None)
                    citation["segments"].append(
                        {
                            "label": chunk.web.title.split(".")[:-1][0],
                            "short_url": resolved_url,
                            "value": chunk.web.uri,
                        }
                    )
                except (IndexError, AttributeError, NameError):
                    # Handle cases where chunk, web, uri, or resolved_map might be problematic
                    # For simplicity, we'll just skip adding this particular segment link
                    # In a production system, you might want to log this.
                    pass
        citations.append(citation)
    return citations
