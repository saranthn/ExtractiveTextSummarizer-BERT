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
from rouge_test import *

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

def run_clusters(content, ratio=0.3, algorithm='kmeans') -> List[str]:
    referenced_data = coreference_handler(content)
    processed_sentences = sentence(content)
    features = create_matrix(processed_sentences)
    hidden_args = cluster_features(features, ratio)
    return [content[j] for j in hidden_args]

R1_p=0.0
R1_r=0.0
R1_f=0.0

R2_p=0.0
R2_r=0.0
R2_f=0.0

RL_p=0.0
RL_r=0.0
RL_f=0.0

for i in range(1,5):
    sentences_summary = run_clusters(data[i]['story'],0.3,'kmeans')
    print(len(data[i]['story']))
    summary = '. '.join(sentences_summary)
    gold_summary = data[i]['highlights']
    score = rouge_score(summary, gold_summary)
    print(score)

    R1_p+=score[0]["rouge-1"]["p"]
    R1_r+=score[0]["rouge-1"]["r"]
    R1_f+=score[0]["rouge-1"]["f"]

    R2_p+=score[0]["rouge-2"]["p"]
    R2_r+=score[0]["rouge-2"]["r"]
    R2_f+=score[0]["rouge-2"]["f"]

    RL_p+=score[0]["rouge-l"]["p"]
    RL_r+=score[0]["rouge-l"]["r"]
    RL_f+=score[0]["rouge-l"]["f"]

print("Average score for this document: \n")
print(" Rouge - 1: ")
print(" precision = "+str(R1_p/len(data)))
print(" recall = "+str(R1_r/len(data)))
print(" F score = "+str(R1_f/len(data)))
print("\n   Rouge - 2: ")
print(" precision = "+str(R2_p/len(data)))
print(" recall = "+str(R2_r/len(data)))
print(" F score = "+str(R2_f/len(data)))
print("\n   Rouge - l: ")
print(" precision = "+str(RL_p/len(data)))
print(" recall = "+str(RL_r/len(data)))
print(" F score = "+str(RL_f/len(data)))
