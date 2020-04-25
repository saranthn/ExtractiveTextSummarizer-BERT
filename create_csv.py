import pickle
from cluster import *
from coreference import *
from sentence_handler import *
from bert import *
import csv

with open('cnn_dataset.pkl', 'rb') as f:
    data = pickle.load(f) 

number_of_summaries = len(data)
print(number_of_summaries)

def retrieve_wcss_bcss(content):
    processed_sentences = sentence(content)
    features = create_matrix(processed_sentences)
    wcss, bcss = get_wcss_bcss(features, 0.3)
    return wcss, bcss

with open("data2.csv", "a", newline='') as f:
    thewriter = csv.writer(f)
    for i in range(5000,10000):
        print(i)
        if len(data[i]['highlights']) == 0 or len(data[i]['story']) == 0:
            continue
        golden_summary_length = len(data[i]['highlights'])
        wcss, bcss = retrieve_wcss_bcss(data[i]['story'])
        thewriter.writerow([wcss, bcss, golden_summary_length])
