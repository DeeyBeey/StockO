from llama_index.core.node_parser import SemanticSplitterNodeParser


def get_semantic_splitter(
    embedding_model,
    buffer_size=1,
    breakpoint_percentile_threshold=95,
):
    return SemanticSplitterNodeParser(
        embed_model=embedding_model,
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
    )


def split_documents(splitter, docs):
    return splitter.get_nodes_from_documents(docs)


def extract_chunk_texts(nodes):
    return [node.get_content() for node in nodes]