from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_ollama import ChatOllama


def get_embedding_models(model_name="BAAI/bge-base-en-v1.5"):
    langchain_embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    llamaindex_embedding_model = LangchainEmbedding(langchain_embedding_model)
    return langchain_embedding_model, llamaindex_embedding_model


def get_llm(model_name="qwen3:8b", temperature=0.2):
    return ChatOllama(
        model=model_name,
        temperature=temperature
    )