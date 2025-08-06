from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embedding_model = HuggingFaceEmbedding(
    model_name='Qwen/Qwen3-Embedding-0.6B', 
    device='cuda'
)

embeddings = embedding_model.get_text_embedding("I love tea!")
print(len(embeddings))
print(embeddings[:5])

