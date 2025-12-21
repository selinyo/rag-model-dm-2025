import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import util, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
from time import perf_counter as timer

import torch


#This file will not embed topics, but rather use the embedded chunks to find the correct topic and also deliver a true/false response. Maybe embedding topics will also help and serve as a pointer, but we have the ID applied in the chunk object itself so we can also find it through there.

embedding_model = HuggingFaceEmbedding(
    model_name='Qwen/Qwen3-Embedding-0.6B', 
#    device='cuda'
#This model was used to embedd the chunks of text.
)
document_embeddings_name = "text_chunks_and_embeddings(1).json" #text_chunks_with_embeddings but used here text_chunks_and_embeddings(1)
print(os.path.exists(document_embeddings_name))

#Load the document and put it as doucment_embeddings
if os.path.exists(document_embeddings_name):
    with open(document_embeddings_name, "r", encoding="utf-8") as json_file:
        document_embeddings = json.load(json_file)

lookup_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2")

def embedd_query(query=str,model=embedding_model):
    query = query.lower().strip()
    embedd_query = model.get_text_embedding(query)
    return embedd_query;
    
    
def retrieve_relevant_resources(embedded_query, embedded_documents, embedding_model,n_resourecs_to_return: int = 8, print_time:bool=True):#8 might be a lot? I am going kind of blind, but I assume with so few documents I might bring atleast half of them being irrrelevant if not more.
#Use cosine similarity since dot product required the same torch float32 which the embedded array is not. PyTorch faults or somethings
    start_timer = timer();
    topic_max_id = 0;
    max_score = 0;
    #Get each document out
    high_scoring_documents = []
    for document in document_embeddings:
        #Apply logic of retrieving ["embedding"] key values and apply cosine similarity, eventually storing the highest values in a list, so we can retrieve the MOST relevant documents.
        vector_array = document["embedding"]
        current_score = cosine_similarity(embedded_query,vector_array)
        if max_score < current_score:
            max_score = current_score
            topic_max_id = document["subject_id"]
            document["cosine_score"] = current_score
            high_scoring_documents.append(document)
    sorted(high_scoring_documents, key=lambda d: d['cosine_score'])
    high_scoring_documents[0:n_resourecs_to_return] ## A very computationally heavy way I know but this is what I need.
    end_time = timer();
    print(f"Time to get score on {len(embedded_documents)} different embedded documents: {end_time-start_timer:.5f} seconds")
    print(f"Highest similarity value: {max_score} pertaining to topic_id {topic_max_id}")
    print(f"The highest relevant documents were: {high_scoring_documents[0:n_resourecs_to_return]}, with n {n_resourecs_to_return} documents to return.")
    
    return high_scoring_documents[0:n_resourecs_to_return];
  

    

def match_topic_to_query(query: str, embedded_topics_path: str) -> str:
    query = query.lower().strip()
    embedded_query = embedding_model.get_text_embedding(query)
    
    retrieve_relevant_resources(embedded_query=embedded_query,)
    
    """ with open(embedded_topics_path, 'r') as file:
        embedded_topics = json.load(file)
        file.close()
    
    topic_names = list(embedded_topics.keys())
    topic_embeddings = np.array([embedded_topics[name] for name in topic_names])

    similarities = cosine_similarity([embedded_query], topic_embeddings)[0]
    match_index = np.argmax(similarities)
    matched_topic = topic_names[match_index]
    """
    
    print("Now returning the best matching topic..")
    print(f"Best matched topic returned: {matched_topic}")
    return matched_topic    

    
query_embeddings = embedd_query("Tachycardia is an ambivalent dichotomy, made of several concering factors that increase risk of damage to heart and body.",embedding_model)

retrieve_relevant_resources(query_embeddings,document_embeddings,lookup_model,4,True)

print()

