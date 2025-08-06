import os
import random;
from tqdm.auto import tqdm;



def text_formatter(text: str) -> str:
    """Perform minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()
    
    return cleaned_text
    # More can be developed here later

def open_and_read_md_files(md_path: str) -> list[dict]:
    with open(md_path, 'r', encoding='utf-8') as file:
        read_text = file.read()  
    pages_and_texts = [] 
    
    for text_section in tqdm(read_text.split(".")):
        text = text_section
        text = text_formatter(text=text)
        pages_and_texts.append({
                                "text_section_char_count": len(text),
                                "text_section_word_count":len(text.split(" ")),
                                "text_section_sentence_count_raw": len(text.split(".")),
                                "text_section_token_count": len(text) / 4,
                                "text": text})
    return pages_and_texts

path = "C:\\Users\\syedw\\Desktop\\WelcomeToUniGirlie\\WorkWorkWork\\DM_AI_25\\DM-i-AI-2025\\emergency-healthcare-rag\\data\\topics\\Brain Death\\Brain Death.md"

pages_and_texts = open_and_read_md_files(md_path=path)


import pandas as pd

df = pd.DataFrame(pages_and_texts)
df.head()

from spacy.lang.en import English

nlp = English()

nlp.add_pipe("sentencizer")

doc = nlp("This is a sentence. This is another sentence. I like an Elephant or multiples of elephants. Something something")

assert len(list(doc.sents)) == 4
print(list(doc.sents))

for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["text"]).sents)
    
    # All sentences should be strings, we must convert from spaCy datatype (whatever that means)
    
    item["sentences"] = [str(sentences) for sentences in item["sentences"]]
    
    # Count sentences
    
    item["text_section_sentence_count_spacy"] = len(item["sentences"])
    
    
num_sentence_chunk_size = 10

# Create a function to split lists of tects recursivelt into our chosen chunk size
#
def split_list(input_list: list, slice_size: int = num_sentence_chunk_size) -> list[list[str]]:
    return[input_list[i:i+slice_size] for i in range(0,len(input_list), slice_size)]

test_list = list(range(25))

split_list(test_list)


# Loop through pages and texts and split sentences into chunks

for item in tqdm(pages_and_texts):
    item["sentence_chunks"] = split_list(input_list=item["sentences"], slice_size=num_sentence_chunk_size)
    item["num_chunks"] = len(item["sentence_chunks"])
    

df = pd.DataFrame(pages_and_texts)
df.describe().round(2)


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

print(len(pages_and_chunks))



print(random.sample(pages_and_chunks, k=4))

# We want a minimum threshold for random chunks, so we filter those with too short chunks 

min_token_length = 30
for row in df[df["chunk_token_count"] <= min_token_length].sample(5).iterrows():
    print("Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentences_chunk"]} ")
 