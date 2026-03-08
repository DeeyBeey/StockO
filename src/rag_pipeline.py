from src.retrieval import retrieve_top_k
from src.prompt_builder import build_rag_prompt


def ask_rag(query, chunk_texts, chunk_embeddings, embed_model, llm, k=2):
    retrieved_chunks = retrieve_top_k(
        query=query,
        chunk_texts=chunk_texts,
        chunk_embeddings=chunk_embeddings,
        embed_model=embed_model,
        k=k
    )

    prompt = build_rag_prompt(query, retrieved_chunks)
    response = llm.invoke(prompt)

    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "prompt": prompt,
        "answer": response.content
    }