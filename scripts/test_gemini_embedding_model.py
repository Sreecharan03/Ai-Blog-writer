# scripts/test_gemini_embedding_model.py
from __future__ import annotations

import os
import math
from pathlib import Path
from dotenv import load_dotenv

EMBED_DIM = 1536

def _load_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-12)


def _flatten_floats(val) -> list[float]:
    out: list[float] = []
    if isinstance(val, list):
        for item in val:
            out.extend(_flatten_floats(item))
    else:
        try:
            out.append(float(val))
        except Exception:
            pass
    return out


def _extract_vec(emb) -> list[float]:
    if emb is None:
        return []
    if isinstance(emb, list):
        return _flatten_floats(emb)
    for attr in ("values", "vector", "embedding"):
        if hasattr(emb, attr):
            v = getattr(emb, attr)
            if isinstance(v, list):
                return _flatten_floats(v)
    if isinstance(emb, dict):
        for k in ("values", "vector", "embedding"):
            v = emb.get(k)
            if isinstance(v, list):
                return _flatten_floats(v)
    return []

def main() -> None:
    _load_env()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")

    import google.generativeai as genai

    genai.configure(api_key=api_key)

    texts = [
        "The cat sat on a mat.",
        "A feline is sitting on a rug.",
        "How do I bake a cake?",
    ]

    model = os.getenv("GEMINI_EMBEDDING_MODEL") or "gemini-embedding-001"
    if not model.startswith("models/"):
        if model in ("gemini-embedding-001", "embedding-001", "text-embedding-004"):
            model = "models/gemini-embedding-001"
        else:
            model = f"models/{model}"

    def _embed_once(content):
        return genai.embed_content(
            model=model,
            content=content,
            output_dimensionality=EMBED_DIM,
        )

    print(f"Calling embed_content with model {model} ...")
    result = _embed_once(texts)

    embs: list = []
    if isinstance(result, dict):
        if "embeddings" in result:
            embs = result.get("embeddings") or []
        elif "embedding" in result:
            embs = [result.get("embedding")]

    if len(embs) != len(texts):
        embs = []
        for t in texts:
            r = _embed_once(t)
            if isinstance(r, dict) and "embedding" in r:
                embs.append(r.get("embedding"))
            elif isinstance(r, dict) and "embeddings" in r:
                arr = r.get("embeddings") or []
                if arr:
                    embs.append(arr[0])

    if not embs:
        raise SystemExit("ERROR: No embeddings returned. Print result to debug:\n" + str(result))

    v0_raw = _extract_vec(embs[0])
    v1_raw = _extract_vec(embs[1])
    v2_raw = _extract_vec(embs[2])

    if not v0_raw:
        raise SystemExit("ERROR: Could not extract embedding vector. Raw embeddings:\n" + str(embs[0])[:1000])

    v0 = v0_raw
    v1 = v1_raw
    v2 = v2_raw

    print(f"OK: got {len(embs)} embeddings")
    print(f"OK: embedding dim: {len(v0)}")
    print("first 5 dims:", [round(x, 6) for x in v0[:5]])

    sim_01 = _cosine(v0, v1)
    sim_02 = _cosine(v0, v2)
    print(f"cosine(text0, text1) = {sim_01:.4f}  (should be higher: similar meaning)")
    print(f"cosine(text0, text2) = {sim_02:.4f}  (should be lower: different topic)")

    print("\nOK: Embedding test PASSED")


if __name__ == "__main__":
    main()
