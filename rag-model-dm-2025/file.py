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

#Is not used
#lookup_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2")

def embedd_query(query=str,model=embedding_model):
    query = query.lower().strip()
    embedd_query = model.get_text_embedding(query)
    return embedd_query;


def retrieve_relevant_resources(embedded_query, embedded_documents,n_resourecs_to_return: int = 8):#8 might be a lot? I am going kind of blind, but I assume with so few documents I might bring atleast half of them being irrrelevant if not more.
#Use cosine similarity since dot product required the same torch float32 which the embedded array is not. PyTorch faults or somethings

    #Avoid the 1D cosine similarity error by making reshaping into a 2D space even though it doesnt change the array at all like what is with these models
    cosine_embedded_query = [embedded_query]

    #Pull out the vector embeddings from the documents.
    vector_array_embeddings = np.array([document["embedding"] for document in embedded_documents])

    #Compute all the cosine similarities for each document instead of going through a loop like last time.
    cosine_similarity_value = cosine_similarity(cosine_embedded_query, vector_array_embeddings)[0]

    #This loop will attach each score to the documents for the query 
    for i in range(0,len(cosine_similarity_value)):
        embedded_documents[i]["cosine_similarity_score"] = cosine_similarity_value[i]
    #Sorting to get the correct ones, reverse = True since we want the highest score
    highest_score_document = sorted(embedded_documents, key=lambda d: d['cosine_similarity_score'],reverse=True)
    return highest_score_document

def print_top_results_and_score(query:str, embedded_documents, embedding_model,n_resourecs_to_return: int = 8):
    start_timer = timer();
    #Embedd the query
    embedded_query =  embedd_query(query=query,model=embedding_model)

    #Retrieve the relevant documents 
    highest_score_document = retrieve_relevant_resources(embedded_query=embedded_query,embedded_documents=embedded_documents,n_resourecs_to_return=4)

    #print them all
    print(f"The highest relevant documents were:... with n {n_resourecs_to_return} documents to return.")
    for x in range(0,n_resourecs_to_return):
        print(f"{highest_score_document[x]["subject_name"]},{highest_score_document[x]["cosine_similarity_score"]}")
    print(f"Highest similarity value: {highest_score_document[0]["cosine_similarity_score"]} pertaining to topic_id {highest_score_document[0]["subject_id"]} name, {highest_score_document[0]["subject_name"]}, document {highest_score_document[0]["file_name"]} with text: {highest_score_document[0]["text"]}")
    end_time = timer()
    print(f"Time to get score on {len(embedded_documents)} different embedded documents: {end_time-start_timer:.5f} seconds")


print_top_results_and_score(query="Pregnancy may lead to higher boneloss and cancer risks, but the best thing for the child is using ultrasounds",embedded_documents=document_embeddings,embedding_model=embedding_model,n_resourecs_to_return=8)
