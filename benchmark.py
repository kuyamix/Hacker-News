# benchmark.py
import os, time, json, logging, sqlite3
from server import state, lifespan, app, _mmr_rerank
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("hn_rag").setLevel(logging.ERROR)

QUERIES = [
    "What are the latest developments in AI?",
    "Is Rust actually better than C++ for systems programming?",
    "How to build a successful startup",
    "Remote work vs return to office",
    "Privacy concerns with modern web browsers",
    "Database recommendations for high traffic",
    "Best mechanical keyboards for programming",
    "Thoughts on the current tech job market",
    "Linux desktop experience in 2024",
    "Self-hosting your own cloud services",
]


def db_conn():
    conn = sqlite3.connect("hn.db")
    conn.row_factory = sqlite3.Row
    return conn


def run_benchmark():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    K_RESULTS = 15

    configs = [
        {"name": "Vector Only (No MMR)", "mmr": 1.0},
        {"name": "Vector + MMR (λ=0.5)", "mmr": 0.5},
    ]

    print(f"\nEvaluating {len(QUERIES)} queries, retrieving {K_RESULTS} raw chunks directly from FAISS...\n")
    print(f"{'Configuration':<25} | {'Avg Time (ms)':<15} | {'Avg Chunk Redundancy':<20} | {'Unique Stories'}")
    print("-" * 85)

    index = state["index"]
    row_to_pair = state["row_to_pair"]

    for cfg in configs:
        total_time = 0
        total_redundancy = 0
        total_unique = 0
        valid_runs = 0

        for q in QUERIES:
            qvec = model.encode([q], normalize_embeddings=True).astype("float32")
            scores, idxs = index.search(qvec, K_RESULTS)

            raw_candidates = []
            relevance = {}
            for score, i in zip(scores[0], idxs[0]):
                if i < 0:
                    continue
                pair = row_to_pair.get(int(i))
                if not pair:
                    continue
                story_id, chunk_idx = pair
                raw_candidates.append(story_id)
                if story_id not in relevance or score > relevance[story_id][0]:
                    relevance[story_id] = (float(score), chunk_idx, "")

            if cfg["mmr"] < 1.0:
                start = time.perf_counter()
                selected_ids = _mmr_rerank(qvec[0], raw_candidates, relevance, K_RESULTS, cfg["mmr"])
                elapsed = (time.perf_counter() - start) * 1000
            else:
                start = time.perf_counter()
                selected_ids = list(dict.fromkeys(raw_candidates))[:K_RESULTS]
                elapsed = (time.perf_counter() - start) * 1000

            real_texts = []
            with db_conn() as conn:
                for sid in selected_ids:
                    cidx = relevance[sid][1]
                    row = conn.execute(
                        "SELECT doc FROM chunks WHERE story_id=? AND chunk_idx=?",
                        (sid, cidx)).fetchone()
                    if row:
                        real_texts.append(row["doc"])

            if not real_texts:
                continue

            vecs = model.encode(real_texts, show_progress_bar=False)
            pairwise = 0
            pairs = 0
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    pairwise += float(vecs[i] @ vecs[j])
                    pairs += 1

            avg_sim = pairwise / pairs if pairs > 0 else 0
            total_time += elapsed
            total_redundancy += avg_sim
            total_unique += len(set(selected_ids))
            valid_runs += 1

        if valid_runs > 0:
            print(f"{cfg['name']:<25} | {total_time/valid_runs:,.1f} ms       | "
                  f"{total_redundancy/valid_runs:.3f}                | "
                  f"{total_unique/valid_runs:,.1f}")


if __name__ == "__main__":
    import asyncio
    async def setup():
        async with lifespan(app):
            run_benchmark()
    asyncio.run(setup())