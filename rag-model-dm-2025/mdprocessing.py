import os
import random;
import spacy;
import en_core_web_sm;
from tqdm.auto import tqdm;

nlp = spacy.load("en_core_web_sm")

nlp = en_core_web_sm.load()

path = "rag-model-dm-2025/data/topics/"

pages_and_texts = [] 

def text_formatter(text: str) -> str:
    """Perform minor formatting on text."""
    cleaned_text = text.replace("\n", " ")
    cleaned_text = cleaned_text.replace("\n\n", " ").strip()
    return cleaned_text
    # More can be developed here later
    

def open_and_read_md_files(md_path: str) -> list[dict]:
    path = "rag-model-dm-2025/data/topics/"
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith((".md")):
                filepath = os.path.join(root,name)
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if "Continuing Education Activity" in content:
                        intro_removed = content.split("Continuing Education Activity")
                    else: 
                        intro_removed = content.split("Introduction")
                    if "## Review Questions" in content:
                        reference_removed = intro_removed[1].split("## Review Questions")
                    else:
                        reference_removed =  intro_removed[1].split("## References")
                    
                    doc = nlp(reference_removed[0])
                   
                    for text_section in tqdm(doc.sents, desc=f"{name}"):
                        text = str(text_section)
                        text = text_formatter(text=text)
                        pages_and_texts.append({
                                                "text_section_char_count": len(text),
                                                "text_section_word_count":len(text.split(" ")),
                                                "text_section_sentence_count_raw": len(text.split(".")),
                                                "text_section_token_count": len(text) / 4,
                                                "text": text})
    return pages_and_texts


pages_and_texts = open_and_read_md_files(md_path=path)


import pandas as pd

df = pd.DataFrame(pages_and_texts)
df.head()


for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["text"]).sents)
    
    # All sentences should be strings, we must convert from spaCy datatype (whatever that means)
    
    item["sentences"] = [str(sentences) for sentences in item["sentences"]]
    
    # Count sentences
    
    item["text_section_sentence_count_spacy"] = len(item["sentences"])
    
    
num_sentence_chunk_size = 10

# Create a function to split lists of tects recursively into our chosen chunk size

def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    return[input_list[i:i+slice_size] for i in range(0,len(input_list), slice_size)]


# Loop through pages and texts and split sentences into chunks

for item in tqdm(pages_and_texts):
    item["sentence_chunks"] = split_list(input_list=item["sentences"], slice_size=num_sentence_chunk_size)
    item["num_chunks"] = len(item["sentence_chunks"])





"""We would like to embed each cuhnk of sentences into its own numerical representation
This will give us a good level granularity. Meaning, we can dive specifically into the text sample that was used in our model"""

import re

pages_and_chunks = []
for item in tqdm(pages_and_texts):
    for sentences_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        
        # Join sentences back into a paragraph from seperate sentence chunks
        
        joined_sentence_chunk = "".join(sentences_chunk).replace("  ","  ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])',r'.\1', joined_sentence_chunk) # .A -> . A (Gives space to capital letters) substitites a selection of capital letters with . with the first char after a .
        
        chunk_dict["sentence_chunk"] = joined_sentence_chunk
        
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4
        
        pages_and_chunks.append(chunk_dict)



# We want a minimum threshold for random chunks, so we filter those with too short chunks 

df = pd.DataFrame(pages_and_chunks)


min_token_length = 8


pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")



    
### Embedding our text chunks with embedding models


from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(model_name_or_path="Qwen/Qwen3-Embedding-0.6B")

# Send the model to the GPU NAHH ITS CPU ONLY IN THIS HOUSE I GOTA CUDA GPU AND I STILL CANNOT RUN THIS SHIT?!
embedding_model.to("cpu") 

# Create embeddings one by one on the GPU
embedding_file = []
for item in tqdm(pages_and_chunks_over_min_token_len):
    item["embedding"] = embedding_model.encode(item["sentence_chunk"])
    embedding_file.append(item)
    

import numpy as np

np.save('embeddings.npy', embedding_file)