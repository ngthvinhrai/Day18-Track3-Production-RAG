"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    
    stat = {
        "Chunks": len(chunks),
        "Avg Len": sum([len(c.text) for c in chunks])/len(chunks),
        "Min": min([len(c.text) for c in chunks]),
        "Max": max([len(c.text) for c in chunks])
    }

    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    # Implement semantic chunking
    # 1. Split text into sentences:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    #
    # 2. Encode sentences:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)
    #
    # 3. Compare consecutive sentences:
    from numpy import dot
    from numpy.linalg import norm
    def cosine_sim(a, b): return dot(a, b) / (norm(a) * norm(b))
    #
    # 4. Group sentences:
    chunks = []
    current_group = [sentences[0]]
    chunk_index = 0
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i-1], embeddings[i])
        if sim < threshold:
            chunks.append(Chunk(text=" ".join(current_group), metadata={"chunk_index": chunk_index, "strategy": "sementic"}))
            current_group = []
            chunk_index += 1
        current_group.append(sentences[i])
    chunks.append(Chunk(text=" ".join(current_group), metadata={"chunk_index": chunk_index, "strategy": "sementic"}))
    

    # 5. Return chunks with metadata: {"chunk_index": i, "strategy": "semantic"}
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────
def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Default recommendation cho production RAG.
    """
    metadata = metadata or {}
    parents: list[Chunk] = []
    children: list[Chunk] = []

    # -------- 1. Split text into parent chunks (by paragraph aggregation) --------
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current_parent_parts = []
    current_len = 0
    parent_index = 0

    def flush_parent():
        nonlocal parent_index, current_parent_parts, current_len
        if not current_parent_parts:
            return None

        parent_text = "\n\n".join(current_parent_parts)
        pid = f"parent_{parent_index}"
        parent_chunk = Chunk(
            text=parent_text,
            metadata={
                **metadata,
                "chunk_type": "parent",
                "parent_id": pid,
            },
        )
        parents.append(parent_chunk)

        parent_index += 1
        current_parent_parts = []
        current_len = 0
        return parent_chunk

    for para in paragraphs:
        if current_len + len(para) > parent_size:
            flush_parent()
        current_parent_parts.append(para)
        current_len += len(para)

    last_parent = flush_parent()

    # -------- 2. Split each parent into child chunks (sliding window) --------
    for parent in parents:
        pid = parent.metadata["parent_id"]
        parent_text = parent.text

        start = 0
        text_len = len(parent_text)

        while start < text_len:
            end = start + child_size
            child_text = parent_text[start:end].strip()
            if child_text:
                child_chunk = Chunk(
                    text=child_text,
                    metadata={
                        **metadata,
                        "chunk_type": "child",
                    },
                    parent_id=pid,
                )
                children.append(child_chunk)

            start += child_size  # no overlap; add overlap here if needed

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    # Implement structure-aware chunking
    # 1. Split by markdown headers:
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    #
    # 2. Pair headers with their content:
    chunks = []
    current_header = ""
    current_content = ""
    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            if current_content.strip():
                chunks.append(Chunk(
                    text=f"{current_header}\n{current_content}".strip(),
                    metadata={**metadata, "section": current_header, "strategy": "structure"}
                ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part
        
    chunks.append(Chunk(
        text=f"{current_header}\n{current_content}".strip(),
        metadata={**metadata, "section": current_header, "strategy": "structure"}
    ))

    #
    # 3. Return chunks — mỗi chunk = 1 section hoàn chỉnh
    #
    # Ưu điểm: giữ nguyên tables, lists, code blocks
    # Dùng khi: corpus có structured documents (docs, API refs, manuals)
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    # Implement comparison
    # 1. For each doc, run: chunk_basic, chunk_semantic, chunk_hierarchical, chunk_structure_aware
    # 2. Collect stats: num_chunks, avg_length, min_length, max_length
    # 3. Print comparison table:
    #    Strategy      | Chunks | Avg Len | Min | Max
    #    basic         |   12   |   420   | 100 | 500
    #    semantic      |    8   |   580   | 200 | 900
    #    hierarchical  | 5p/15c |   256   | 100 | 2048
    #    structure     |   10   |   450   | 150 | 800
    # 4. Return results dict

    def get_stat(chunks):
        lengths = [len(c.text) for c in chunks]
        return {
            "num_chunks": len(chunks),
            "avg_length": sum(lengths)/len(chunks),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }


    results = {
        "basic": [],
        "semantic": [],
        "hierarchical_parents": [],
        "hierarchical_children": [],
        "structure": [],
    }

    for doc in documents:

        text = doc["text"]
        metadata = doc.get("metadata", {})

        results["basic"].extend(
            chunk_basic(text, metadata=metadata)
        )

        results["semantic"].extend(
            chunk_semantic(text, metadata=metadata)
        )

        parents, children = chunk_hierarchical(
            text, metadata=metadata
        )
        results["hierarchical_parents"].extend(parents)
        results["hierarchical_children"].extend(children)

        results["structure"].extend(
            chunk_structure_aware(text, metadata=metadata)
        )


    stats = {
        "basic": get_stat(results["basic"]),
        "semantic": get_stat(results["semantic"]),
        "hierarchical": {
            "parents": get_stat(results["hierarchical_parents"]),
            "children": get_stat(results["hierarchical_children"]),
        },
        "structure": get_stat(results["structure"]),
    }


    return stats


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
