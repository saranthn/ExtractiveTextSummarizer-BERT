import pickle

with open('cnn_dataset.pkl', 'rb') as f:
    data = pickle.load(f) 

from sklearn.cluster import KMeans
from typing import List
import numpy as np
from numpy import ndarray
import torch
from transformers import BertTokenizer, BertModel, BertConfig
import logging
from cluster import *
from sentence_handler import *
from coreference import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

def extract_embeddings(sentence) -> ndarray:
    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(sentence)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Print out the tokens.
    #print (tokenized_text)

    # Display the words with their indices.
    #for tup in zip(tokenized_text, indexed_tokens):
    #    print('{:<12} {:>6,}'.format(tup[0], tup[1]))
        
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        outputs = model(tokens_tensor)

    #Sequence of hidden-states at the last layer of the model.
    #torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
    last_hidden_states = outputs[0]

    #Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. 
    #This output is usually not a good summary of the semantic content of the input, It is better with averaging or pooling the sequence of hidden-states for the whole input sequence.
    #torch.FloatTensor: of shape (batch_size, hidden_size)
    pooler_output = outputs[1]

    #Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    #torch.FloatTensor (one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size)
    hidden_states = outputs[2]

    embedding_output = hidden_states[0]
    attention_hidden_states = hidden_states[1:]

    pooled = hidden_states[-2].mean(dim=1)

    return pooled

def create_matrix(content) -> ndarray:
    return np.asarray([
        np.squeeze(extract_embeddings(t).data.numpy())
        for t in content
    ])

def run_clusters(content, ratio=0.5, algorithm='kmeans') -> List[str]:
    print(content)
    referenced_data = coreference_handler(content)
    print(referenced_data)
    processed_sentences = sentence(referenced_data)
    print(processed_sentences)
    features = create_matrix(processed_sentences)
    hidden_args = cluster_features(features, ratio)
    return [content[j] for j in hidden_args]

sentences_summary = run_clusters(data[876]['story'],0.5,'kmeans')
summary = ' '.join(sentences_summary)
print(summary)