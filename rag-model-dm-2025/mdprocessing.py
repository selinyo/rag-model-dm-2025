import json
import os
import random; #Why were u here????
import spacy;
import en_core_web_sm;
from tqdm.auto import tqdm;

nlp = spacy.load("en_core_web_sm")

nlp = en_core_web_sm.load()

path = "rag-model-dm-2025/data/topics/"
    
pages_and_texts = [] 

with open("rag-model-dm-2025/data/topics.json",encoding="utf-8") as file:
    topics_to_ids = json.load(file)

print(topics_to_ids)
def text_formatter(text: str) -> str:
    """Perform minor formatting on text."""
    cleaned_text = text.replace("\n", " ")
    cleaned_text = cleaned_text.replace("\n\n", " ").strip()
    return cleaned_text
    # More can be developed here later
        # No, I refuse.
    
num_sentence_chunk_size = 10

# Create a function to split lists of tects recursively into our chosen chunk size
def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    return[input_list[i:i+slice_size] for i in range(0,len(input_list), slice_size)]

def open_and_read_md_files(md_path: str) -> list[dict]:

    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith((".md")):
                filepath = os.path.join(root,name)
                folder_name = os.path.basename(root)
                folder_id = topics_to_ids.get(folder_name, 404) #Get it 404, cause like it wouldnt find haha funny compsi joke please laugh im actually going insane we are a duo but it feels like we are silvers landing in a GM lobby
                with open(filepath, 'r', encoding='utf-8') as file:
                    print(name)
                    content = file.read()
                    ##Removing redundant stuff like refrences and intro with authors
                    if "Continuing Education Activity" in content:
                        intro_removed = content.split("Continuing Education Activity")
                    else: 
                        intro_removed = content.split("Introduction")
                    if "## Review Questions" in content:
                        reference_removed = intro_removed[1].split("## Review Questions")
                    else:
                        reference_removed =  intro_removed[1].split("## References")
                    
                    doc = nlp(reference_removed[0])
                   
                    sentences = [text_formatter(str(sent)) for sent in doc.sents]
                    
                    sentence_chunks = split_list(sentences, slice_size=num_sentence_chunk_size)
                    
                    for text in sentence_chunks:
                        clean_text = " ".join(text).strip()
                        pages_and_texts.append({"subject_name": folder_name,
                                                "subject_id": folder_id, #I forgot this last time so the data was practically useless as we had no way to confirm which document it came from GG WP dumbo
                                                "file_name": name,
                                                "text_section_char_count": len(clean_text),
                                                "text_section_word_count":len(clean_text.split(" ")),
                                                "text_section_sentence_count_raw": len(clean_text.split(".")),
                                                "text_section_token_count": len(clean_text) / 4, #Why four, cause its perfection
                                                "text": clean_text,
                                                "sentence_chunks": text})
    
    return pages_and_texts

output_json_path = "processed_text_chunks.json"

#We dont need to make the same extraction file every goddamn time and waste MORE TIME THANS I ALREADY HAVE
if os.path.exists(output_json_path):
    with open(output_json_path, "r", encoding="utf-8") as json_file:
        pages_and_texts = json.load(json_file)
else:
    pages_and_texts = open_and_read_md_files(md_path=path)
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(pages_and_texts, json_file, ensure_ascii=False, indent=2)
    
import pandas as pd

df = pd.DataFrame(pages_and_texts)
print(df.head())



min_token_length = 8 #Honestly you were just random, maybe I should have done u better but its like two words so i beleive its fine

pages_and_chunks_over_min_token_len = df[df["text_section_token_count"] > min_token_length].to_dict(orient="records")

print(df.sample(6)) # Just to see, if u know.. things are bad 



    
### Embedding our text chunks with embedding models


from sentence_transformers import SentenceTransformer
# Load the model
embedding_model = SentenceTransformer(model_name_or_path="Qwen/Qwen3-Embedding-0.6B")

# Extract texts and ONLY texts
text_chunks = [item["text"] for item in pages_and_chunks_over_min_token_len]


text_chunk_embeddings = embedding_model.encode(
    text_chunks,
    batch_size=16, #I CANT HANDLE MORE :SOB_EMOJI:
    show_progress_bar=True,
    convert_to_tensor=False 
)

# Attach embeddings to each item and convert to list for JSON serialization
for item, emb in zip(pages_and_chunks_over_min_token_len, text_chunk_embeddings):
    # Convert the embedding to a list and attach it
    item["embedding"] = emb.tolist()  # Embedding as a list so we don't truncate data like last time and start crying when the model breaks.

# Now save the entire data (texts + embeddings) to a JSON file, cause again I refuse to lose this file.
output_json_path = "text_chunks_with_embeddings.json"

# Simple write function
with open(output_json_path, "w", encoding="utf-8") as json_file:
    json.dump(pages_and_chunks_over_min_token_len, json_file, ensure_ascii=False, indent=2)

