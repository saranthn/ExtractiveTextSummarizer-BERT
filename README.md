# Extractive Text Summarization using BERT
<a name="readme-top"></a>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#how-to-use">How to use</a></li>
      </ul>
    </li>
    <li><a href="#running-the-tests">Running the tests</a></li>
    <li><a href="#contributors">Contributors</a></li>
    <li><a href="#reference">Reference</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The process of picking sentences directly from the story to form the summary is extractive summarization. This process is aided by scoring functions and clustering algorithms to help choose the most suitable sentences. We use the existing BERT model which stands for Bidirectional Encoder Representations from Transformers, to produce extractive summarization by clustering the embeddings of sentences by K-means clustering, but introduce a dynamic method to decide the suitable number of sentences to pick from clusters.On top of that, the study is aimed at producing summaries of higher quality by incorporating reference resolution and dynamically producing summaries of suitable sizes depending on the text. This study aims to provide students with a summarizing service to help understand the content of lecture videos of long duration which would be vital in the process of revision.

This repository contains an extractive summarization tool that uses BERT to produce embeddings for clustering using a K-Means model to produce the summary. This is tested in CNN_Dailymail dataset

[Publication](https://ieeexplore.ieee.org/abstract/document/9277220)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

* ![Python]
* [Transformers](https://github.com/huggingface/transformers) - BERT model
* [NeuralCoref](https://github.com/huggingface/neuralcoref) - Reference Resolution
* [Spacy](https://spacy.io/api/doc) - Used for NLP language

<p align="right">(<a href="#readme-top">back to top</a>)</p>

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
<p align="right">(<a href="#readme-top">back to top</a>)</p>

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

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Running the tests

Set the lower limit and upper limit of the CNN dataset to run the test in summarize.py

```
python3 summarize.py test_cnn
```
The result is the average ROUGE-1, ROUGE-2 and ROUGE-l score across the range of documents specified 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributors

* **Anirudh S** - (https://github.com/anirudhs1998)
* **Saravanan T** - (https://github.com/saranthn)
* **Ashwin Shankar** - (https://github.com/Ashwinshankar98)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Reference

Paper: https://arxiv.org/abs/1906.04165

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[Python]: https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white

