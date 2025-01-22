
# Sentiment Analysis with Keras
## Overview
This project explores sentiment analysis using a custom dataset (`moviereviews2.tsv`) and compares the performance of two approaches:  
1. **NLTK's VADER Sentiment Analysis**: A lexicon-based approach.  
2. **A Deep Learning Model Built with Keras**: A neural network trained on vectorized movie review texts.

### Motivation
The dataset was introduced in a Udemy NLP course and originally analyzed using VADER. However, VADER's performance was suboptimal for this dataset, prompting an investigation into alternative methods. The hypothesis was that a deep learning model could outperform VADER by learning patterns in the text through Tensorflow tokenization.

## Dataset
The dataset, `moviereviews2.tsv`, contains movie reviews labeled as positive or negative. Each review serves as an input, and its sentiment label serves as the target for training and evaluation.

- **Columns**:
  - `review`: The text of the movie review.
  - `label`: The sentiment label (`0` for negative, `1` for positive).

## Methodology

### 1. Preprocessing the Data
- Removed null values and duplicates.
- Split the data into training and testing sets.
- Tokenized the review texts using Keras's Tensorflow `Tokenizer` to prepare them for the model.

### 2. VADER Sentiment Analysis
- Applied VADER on the dataset.
- Measured accuracy and observed its limitations in handling nuanced or complex reviews.

### 3. Building the Keras Model
- Designed a Sequential model with embedding layers to process the tokenized text.
- Used the following architecture:
  - Embedding layer
  - LSTM/GRU layer
  - Dense output layer with sigmoid activation
- Trained the model on the dataset and evaluated its accuracy.

## Results
- **VADER**: Achieved a disappointing accuracy of ~63.67%.
- **Keras Model**: Significantly outperformed VADER with an accuracy of approximately `86.29%`.

## Key Features
- Utilizes deep learning to improve sentiment prediction
