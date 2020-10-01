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

python -m spacy download en_core_web_md
```

### How to use

Pre-process the dataset 

```
python3 preprocess.py
```

This will produce cnn_dataset.pkl.

To produce the summary of any document, set the document in summarize.py

```
python3 summarize.py create_summary
```
To produce the summary of any specific CNN document, specify the document number(0-90000) in summarize.py

```
python3 summarize.py create_summary_cnn_single
```

To collect **within cluster sum of squares**, **between cluster sum of squares**, **summary length** data

```
python3 summarize.py collect_data
```

To train a linear regression model between **within cluster sum of squares** and **summary length**

```
python3 summarize.py train_model
```

To check the histogram produced by **within cluster sum of squares**

```
python3 histogram_wcss
```

## Running the tests

Set the lower limit and upper limit of the CNN dataset to run the test in summarize.py

```
python3 summarize.py test_cnn
```
The result is the average ROUGE-1, ROUGE-2 and ROUGE-l score across the range of documents specified 

## Built With

* [Transformers](https://github.com/huggingface/transformers) - BERT model
* [NeuralCoref](https://github.com/huggingface/neuralcoref) - Reference Resolution
* [Spacy](https://spacy.io/api/doc) - Used for NLP language
 

## This Work has been published in the ICCCS IEEE conference at IIT Patna on October 14-16, 2020.

Link to our research paper: Soon !!

## Contributors

* **Anirudh S** - (https://github.com/anirudhs1998)
* **Saravanan T** - (https://github.com/saranthn)
* **Ashwin Shankar** - (https://github.com/Ashwinshankar98)

## Reference

Paper: https://arxiv.org/abs/1906.04165

