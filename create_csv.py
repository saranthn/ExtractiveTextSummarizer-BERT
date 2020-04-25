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
    referenced_data = coreference_handler(content)
    processed_sentences = sentence(referenced_data)
    features = create_matrix(processed_sentences)
    wcss, bcss = get_wcss_bcss(features, 0.3)
    return wcss, bcss

with open("data.csv", "w", newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(['wcss','bcss','length_of summary'])
    for i in range(1,4):
        print(i)
        golden_summary_length = len(data[i]['highlights'])
        wcss, bcss = retrieve_wcss_bcss(data[i]['story'])
        thewriter.writerow([wcss, bcss,golden_summary_length])
