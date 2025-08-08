from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np

embedding_model = HuggingFaceEmbedding(
    model_name='Qwen/Qwen3-Embedding-0.6B', 
#    device='cuda'
)

def embed_topics(topic_path: str, embedded_topics_path: str) -> None:
    with open(topic_path, 'r') as file1:
        topics = json.load(file1)
        file1.close()
    with open(embedded_topics_path, 'w') as file2:
        embedded_topics = {}
        for topic in topics:
            embedded_topics[topic] = embedding_model.get_text_embedding(topic)
        json_str = json.dumps(embedded_topics, indent=4)
        file2.write(json_str)
        file2.close()
        
def match_topic_to_query(query: str, embedded_topics_path: str) -> str:
    query = query.lower().strip()
    embedded_query = embedding_model.get_text_embedding(query)

    with open(embedded_topics_path, 'r') as file:
        embedded_topics = json.load(file)
        file.close()
    
    topic_names = list(embedded_topics.keys())
    topic_embeddings = np.array([embedded_topics[name] for name in topic_names])

    similarities = cosine_similarity([embedded_query], topic_embeddings)[0]
    match_index = np.argmax(similarities)
    matched_topic = topic_names[match_index]
    print("Now returning the best matching topic..")
    print(f"Best matched topic returned: {matched_topic}")
    return matched_topic

print("hei")
embed_topics("C:/Users/selin/Documents/annet/hackaton/rag-model-dm-2025/rag-model-dm-2025/data/topics.json", "C:/Users/selin/Documents/annet/hackaton/rag-model-dm-2025/rag-model-dm-2025/data/embedded_topics.json")
match_topic_to_query("Subtotal cholecystectomy may be performed when severe pericholecystic inflammation makes safe dissection of Calot's triangle impossible, though this approach is rarely necessary in empyema cases.", "C:/Users/selin/Documents/annet/hackaton/rag-model-dm-2025/rag-model-dm-2025/data/embedded_topics.json")
print("hadet, funket det ikke?")

# for x in range(0, 200):
#     statement = "statement_0{number}.txt".format(number = x)
#     answer = "statement_0{number}.json".format(number = x)
