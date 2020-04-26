import pickle

with open('cnn_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

import logging
from cluster import *
from sentence_handler import *
from rouge import Rouge

#from coreference import *
from coreference import *
from rouge_test import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def create_matrix(content) -> ndarray:
        for t in content
    ])

def run_clusters(content, ratio=0.5, algorithm='kmeans') -> List[str]:
    print(content)
    #referenced_data = coreference_handler(content)
    #print(referenced_data)
def run_clusters(content, ratio=0.3, algorithm='kmeans') -> List[str]:
    referenced_data = coreference_handler(content)
    processed_sentences = sentence(content)
    print(processed_sentences)
    features = create_matrix(processed_sentences)
    hidden_args = cluster_features(features, ratio)
    return [content[j] for j in hidden_args]

#sentences_summary = run_clusters(data[876]['story'],0.5,'kmeans')
#summary = ' '.join(sentences_summary)


for i in range(1,5):
    sentences_summary = run_clusters(data[i]['story'],0.3,'kmeans')
    print(len(data[i]['story']))
    summary = '. '.join(sentences_summary)
    gold_summary = data[i]['highlights']
    score = rouge_score(summary, gold_summary)
    print(score)
