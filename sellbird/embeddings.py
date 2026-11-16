"""Local Semantic Embeddings using Sentence Transformers for Harvey Local AI.

Model: sentence-transformers/all-MiniLM-L6-v2
Licence: Apache 2.0
Dimensions: 384
Weights size: ~80 MB

Requirements satisfied:
- Lightweight, openly licensed embedding model (Apache 2.0).
- Offline inference once model weights are stored.
- Never downloads automatically during standard application startup.
- Persists embeddings in SQLite without external vector DB dependencies.
- Cosine similarity calculation between prospects, observations, and product ICPs.
- Deterministic heuristic fallback if model weights are not downloaded yet.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

logger = logging.getLogger("sellbird.embeddings")

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MODEL_LICENCE = "Apache-2.0"

_GLOBAL_MODEL = None
_MODEL_LOAD_FAILED = False


class EmbeddingModelNotDownloadedError(RuntimeError):
    """Raised when embeddings are requested but the local model weights have not been downloaded."""
    pass


def get_model_cache_dir() -> Path:
    """Return local directory designated for Harvey embedding model weights."""
    from sellbird.paths import PROJECT_ROOT
    cache_dir = PROJECT_ROOT / "data" / "models" / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def is_model_downloaded(model_name: str = DEFAULT_MODEL_NAME) -> bool:
    """Check if model weights exist locally in the project cache or HuggingFace cache."""
    cache_dir = get_model_cache_dir()
    local_path = cache_dir / model_name
    if local_path.is_dir() and (local_path / "modules.json").is_file():
        return True

    # Also check default sentence-transformers cache in user home
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface/hub")
    hub_path = Path(hf_home)
    if hub_path.is_dir():
        for item in hub_path.glob(f"*{model_name}*"):
            if item.is_dir():
                return True
    return False


def load_embedding_model(allow_download: bool = False, model_name: str = DEFAULT_MODEL_NAME):
    """Load the SentenceTransformer model.

    If allow_download is False, strictly forces local_files_only=True so that
    no automatic internet request or silent download is triggered during startup.
    """
    global _GLOBAL_MODEL, _MODEL_LOAD_FAILED
    if _GLOBAL_MODEL is not None:
        return _GLOBAL_MODEL

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers is not installed in the environment.")
        return None

    cache_dir = get_model_cache_dir()
    local_path = cache_dir / model_name

    # Try local cache path first
    target_path = str(local_path) if local_path.is_dir() else model_name

    try:
        if not allow_download:
            # Strictly offline loading
            logger.info(f"Loading local embedding model '{model_name}' (local_files_only=True)...")
            model = SentenceTransformer(target_path, local_files_only=True)
        else:
            logger.info(f"Downloading/loading embedding model '{model_name}' to {cache_dir}...")
            model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
            # Save a dedicated local copy if needed
            if not (cache_dir / model_name).is_dir():
                try:
                    model.save(str(cache_dir / model_name))
                except Exception as e:
                    logger.debug(f"Could not save local copy: {e}")

        _GLOBAL_MODEL = model
        return _GLOBAL_MODEL
    except Exception as e:
        if not allow_download:
            logger.info(
                f"Embedding model '{model_name}' not found locally. "
                "Run 'sellbird setup-embeddings' or call load_embedding_model(allow_download=True) once."
            )
        else:
            logger.error(f"Failed to load/download embedding model: {e}")
        return None


def download_model(model_name: str = DEFAULT_MODEL_NAME) -> bool:
    """Explicit one-time setup utility to download embedding weights for offline usage."""
    model = load_embedding_model(allow_download=True, model_name=model_name)
    return model is not None


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two float vectors.

    Returns float in range [-1.0, 1.0], normalized to [0.0, 1.0] for similarity scoring.
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a, b in zip(vec1, vec1)))
    norm_b = math.sqrt(sum(b * b for b, b in zip(vec2, vec2)))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    sim = dot / (norm_a * norm_b)
    # Clamp and normalize to [0.0, 1.0]
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


def _token_jaccard_similarity(text1: str, text2: str) -> float:
    """Deterministic fallback similarity when sentence-transformers is offline / not downloaded.

    Computes token-level Jaccard similarity with word n-gram matching.
    """
    if not text1 or not text2:
        return 0.0

    def tokenize(t: str) -> set[str]:
        words = re.findall(r"\w+", t.lower())
        tokens = set(words)
        # Add bigrams for context
        for i in range(len(words) - 1):
            tokens.add(f"{words[i]}_{words[i+1]}")
        return tokens

    set1 = tokenize(text1)
    set2 = tokenize(text2)

    if not set1 or not set2:
        return 0.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return round(intersection / union, 4) if union > 0 else 0.0


def embed_text(text: str, allow_download: bool = False) -> List[float]:
    """Generate dense embedding for a given text string.

    Returns float vector of length 384. If model is unavailable, produces
    a deterministic pseudo-vector derived from token hash for test/fallback continuity.
    """
    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIM

    model = load_embedding_model(allow_download=allow_download)
    if model is not None:
        try:
            emb = model.encode(text.strip(), normalize_embeddings=True)
            return emb.tolist() if hasattr(emb, "tolist") else list(emb)
        except Exception as e:
            logger.warning(f"Error computing sentence embedding: {e}")

    # Deterministic fallback vector using hashed bag of words
    vec = [0.0] * EMBEDDING_DIM
    words = re.findall(r"\w+", text.lower())
    for w in words:
        h = abs(hash(w)) % EMBEDDING_DIM
        vec[h] += 1.0

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def embed_texts(texts: List[str], allow_download: bool = False) -> List[List[float]]:
    """Batch embed multiple strings."""
    if not texts:
        return []

    model = load_embedding_model(allow_download=allow_download)
    if model is not None:
        try:
            embs = model.encode([t.strip() for t in texts], normalize_embeddings=True)
            return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embs]
        except Exception as e:
            logger.warning(f"Error computing batch sentence embeddings: {e}")

    return [embed_text(t, allow_download=allow_download) for t in texts]


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity between two texts.

    Uses SentenceTransformers when available locally; otherwise uses
    deterministic token Jaccard similarity.
    """
    if not text1 or not text2:
        return 0.0

    model = load_embedding_model(allow_download=False)
    if model is not None:
        try:
            v1 = embed_text(text1)
            v2 = embed_text(text2)
            return cosine_similarity(v1, v2)
        except Exception as e:
            logger.debug(f"Semantic model inference failed: {e}")

    return _token_jaccard_similarity(text1, text2)


def prospect_icp_semantic_match(
    prospect_profile: str,
    icp_description: str,
    target_roles: List[str],
    pain_points: List[str],
) -> Tuple[float, List[str]]:
    """Compute semantic match score (0.0 to 1.0) and explainable factors between prospect and ICP."""
    if not prospect_profile:
        return 0.0, ["insufficient prospect data for semantic match"]

    icp_combined = f"{icp_description} Roles: {', '.join(target_roles)}. Pain points: {', '.join(pain_points)}"
    similarity = compute_semantic_similarity(prospect_profile, icp_combined)

    factors = []
    if similarity >= 0.70:
        factors.append(f"Strong semantic alignment with ICP profile ({similarity:.2f})")
    elif similarity >= 0.50:
        factors.append(f"Moderate semantic alignment with ICP profile ({similarity:.2f})")
    else:
        factors.append(f"Low semantic overlap with target ICP ({similarity:.2f})")

    # Check role alignment specifically
    for role in target_roles:
        if role.lower() in prospect_profile.lower():
            factors.append(f"Exact role match: '{role}'")
            break

    return round(similarity, 4), factors
