# Sentiment Analysis Pipeline

This repository contains a complete NLP pipeline for processing the wellknown Glassdoor company reviews and training a logistic regression model to classify sentiments. 
It uses a structured process for cleaning text data, removing stopwords, lemmatizing, and extracting features through TF-IDF vectorization. 
Model metrics and visualizations are logged via MLflow.

## Overview

The main script executes the following steps:

1. **Load Dataset**  
   Load the dataset containing company reviews with fields like `headline`, `pros`, and `cons`.

2. **Data Cleaning**  
   - Remove rows with empty values in text fields.
   - Filter out irrelevant categories (e.g., `recommend == 'o'`).
   - Normalize text: lowercase, remove numbers, punctuation, and extra spaces.

3. **Preprocessing**
   - **Stopwords removal**
   - **Lemmatization**
   - **Merge text columns** into a single column for vectorization.

4. **Label Mapping**  
   Map categorical labels in the `recommend` column into ranked integers.

5. **Feature Extraction**  
   Use **TF-IDF vectorization** to convert text into numerical features.

6. **Train/Test Split**  
   Split the dataset into training and testing subsets.

7. **Model Training**  
   Train a **Logistic Regression** model.

8. **Evaluation and Metrics**  
   Evaluate the model and generate predictions and probability scores such as:
   - Accurracy
   - F1-score
   - AUC score
   - ROC curve
   - Confusion matrix
   - Word frequency plots  
   - Bigrams and trigrams frequency plots

9. **MLflow Integration**  
   Log all metrics, model, and visualizations using MLflow for experiment tracking.

---

## Requirements

You can install all required packages with:

```bash
pip install -r requirements.txt
