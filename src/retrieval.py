import numpy as np


def embed_chunks(embedding_model, chunk_texts):
    return embedding_model.embed_documents(chunk_texts)


def embed_query(embedding_model, query):
    return embedding_model.embed_query(query)


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_similarities(query_embedding, chunk_embeddings):
    return [
        cosine_similarity(query_embedding, emb)
        for emb in chunk_embeddings
    ]


def retrieve_top_k(query, chunk_texts, chunk_embeddings, embed_model, k=2):
    query_embedding = embed_query(embed_model, query)

    scores = compute_similarities(query_embedding, chunk_embeddings)

    ranked_results = sorted(
        list(enumerate(scores)),
        key=lambda x: x[1],
        reverse=True
    )

    top_results = ranked_results[:k]

    retrieved_chunks = []
    for idx, score in top_results:
        retrieved_chunks.append({
            "chunk_id": idx + 1,
            "score": score,
            "text": chunk_texts[idx]
        })

    return retrieved_chunks