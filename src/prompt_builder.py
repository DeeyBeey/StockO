def build_rag_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = []

    for item in retrieved_chunks:
        context_blocks.append(
            f"[Chunk {item['chunk_id']} | similarity={item['score']:.4f}]\n{item['text']}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a short-term stock trader.

Use ONLY the context below to answer the question.
You MUST choose either BULLISH or BEARISH. No neutral answers.

Question:
{query}

Context:
{context}

Respond in this format:

SIGNAL: BULLISH or BEARISH

THESIS:
1-3 sentences explaining the view.

EVIDENCE:
- key point from context
- key point from context

RISKS:
- risk to this view

CONFIDENCE: Low / Medium / High
""".strip()

    return prompt