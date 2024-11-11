import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load the dataset
data = pd.read_csv("C:\\Users\\Research_Software_Engineer_Task\\collection_with_abstracts.csv")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Convert Abstract column to strings, replacing NaN with empty strings
data['Abstract'] = data['Abstract'].fillna('').astype(str)
# Define the query and embeddings for semantic filtering
query = "deep learning in virology or epidemiology"
query_embedding = model.encode(query, convert_to_tensor=True)


def filter_relevant_papers(row):
    # Check title + abstract relevance by semantic similarity
    text = f"{row['Title']} {row['Abstract']}"
    text_embedding = model.encode(text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(query_embedding, text_embedding).item()
    return similarity > 0.4  # Threshold for relevance


# Filter dataset for relevant papers
data['is_relevant'] = data.apply(filter_relevant_papers, axis=1)
relevant_papers = data[data['is_relevant']]

# Define category descriptions for method classification
category_descriptions = {
    "text mining": "natural language processing, text mining, NLP, text analysis, computational linguistics, textual data analysis, speech and language technology, language modeling, computational semantics",
    "computer vision": "image recognition, computer vision, segmentation, image processing, vision model, vision algorithms, computer graphics and vision, object recognition, scene understanding",
    "both": "text mining and computer vision combined",
    "other": "deep learning applications not specific to text mining or computer vision"
}
# Encode the descriptions as category embeddings
category_embeddings = {category: model.encode(description) for category, description in category_descriptions.items()}


# Function to classify method type based on similarity
def classify_method(abstract):
    # Handle empty abstracts
    if not abstract.strip():
        return "other"
    abstract_embedding = model.encode(abstract)
    
    # Calculate similarity between abstract and each category description
    similarities = {category: util.pytorch_cos_sim(abstract_embedding, embedding).item()
                    for category, embedding in category_embeddings.items()}
    # Return the category with the highest similarity score
    return max(similarities, key=similarities.get)


relevant_papers['method_type'] = relevant_papers['Abstract'].apply(classify_method)
# Define method keywords
method_keywords = [
    "convolutional neural network", "transformer", "LSTM", "recurrent neural network", "CNN", "RNN", "GAN", "autoencoder"]
method_embeddings = model.encode(method_keywords, convert_to_tensor=True)


# Define a function to extract methods using semantic similarity
def extract_methods_semantically(text):
    # Check if text is non-empty
    if pd.isna(text) or not isinstance(text, str):
        return "Not specified"
    
    # Encode the abstract text
    text_embedding = model.encode(text, convert_to_tensor=True)
    # Calculate similarity with each method keyword
    similarities = [util.pytorch_cos_sim(text_embedding, method_embedding).item() for method_embedding in method_embeddings]
    # Find the methods that exceed a similarity threshold
    threshold = 0.3
    max_score = max(similarities)
    
    if max_score > threshold:
        # Get the index of the method with the maximum score
        best_method_index = similarities.index(max_score)
        return method_keywords[best_method_index]
    else:
        return "Not specified"    


relevant_papers['method_name'] = relevant_papers['Abstract'].apply(extract_methods_semantically)
method_type_counts = relevant_papers['method_type'].value_counts()
method_name_counts = relevant_papers['method_name'].value_counts()
# Display results
print("Method Type Counts:\n", method_type_counts)
print("\nMethod Name Counts:\n", method_name_counts)

# Save to CSV
relevant_papers[['PMID', 'Title', 'Abstract', 'is_relevant', 'method_type', 'method_name']].to_csv("filtered_virology_papers.csv", index=False)