# Extractive Summarization using BERT
This repository contains an extractive summarization tool that uses BERT to produce embeddings for clustering using a K-Means model to produce the summary. This is tested in CNN_Dailymail dataset

## Getting Started

To setup the project, first download the CNN-Dailymail dataset

### Prerequisites

Install necessary packages

```
pip install spacy
pip install transformers
pip install neuralcoref
```

### Installing

Pre-process the dataset 

```
python3 preprocess.py
```

This will produce cnn_dataset.pkl. Now run the bert model and cluster produce summary

```
python3 bert.py
```

## Running the tests

Coming soon!!

## Built With

* [Transformers](https://github.com/huggingface/transformers) - BERT model
* [NeuralCoref](https://github.com/huggingface/neuralcoref) - Reference Resolution
* [Spacy](https://spacy.io/api/doc) - Used for NLP language
 

## Authors

* **Anirudh S** - (https://github.com/anirudhs1998)
* **Saravanan T** - (https://github.com/saranthn)
* **Ashwin Shankar** - (https://github.com/Ashwinshankar98)



