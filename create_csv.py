import pickle

with open('cnn_dataset.pkl', 'rb') as f:
    data = pickle.load(f) 

number_of_summaries = len(data)
print(number_of_summaries)

f = open("data.csv","a")

for i in range(number_of_summaries):
    golden_summary_length = len(data[i]['highlights'])
    print(golden_summary_length)