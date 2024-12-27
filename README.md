### Detection of Hate Speech on Turkish Tweets Using Machine Learning

This repository contains a project focused on detecting hate speech in Turkish tweets using various machine learning techniques. Below, you will find a summary of the steps involved, the methodology applied, and the results achieved.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Data Preprocessing](#data-preprocessing)  
3. [Text Correction](#text-correction)  
4. [Fine-Tuning Turkish Word2Vec Model](#fine-tuning-turkish-word2vec-model)  
5. [Model Training and Evaluation](#model-training-and-evaluation)  
6. [Class Imbalance Handling](#class-imbalance-handling)  
7. [Results and Observations](#results-and-observations)  
8. [Future Work](#future-work)  

---

## Introduction

This project aims to detect hate speech in Turkish tweets using machine learning. It includes data preprocessing, text correction, generating word embeddings, and training/evaluating multiple models. Fine-tuning techniques, Turkish Word2Vec, and resampling methods are employed to improve the results.

---

## Data Preprocessing

- **Data Source:** Tweets were collected and stored in `data.xlsx`.  
- **Steps Performed:**  
  - Removed unnecessary characters, symbols, and URLs.  
  - Tokenized tweets into individual words.  
  - Eliminated stopwords to enhance informativeness.  
  - Applied lemmatization for text normalization.  
- **Output:** The preprocessed data was saved as `data_cleaned.csv`.

---

## Text Correction

- **Tool Used:** Zemberek Library ([Reference 1](https://github.com/ozturkberkay/Zemberek-PythonExamples/blob/master/examples/normalization/spell_checking.py), [Reference 2](https://github.com/ahmetaa/zemberek-nlp/tree/master/normalization)).  
- **Process:**  
  - Morphological analysis using `TurkishMorphology`.  
  - Spelling corrections via `TurkishSpellChecker`.  
- **Result:** Added a `corrected_tweet` column to the dataset.  
- **Output:** The corrected data was saved as `data_cleaned_with_corrections.csv`.

---

## Fine-Tuning Turkish Word2Vec Model

- **Pretrained Model:** [Turkish Word2Vec](https://github.com/akoksal/Turkish-Word2Vec).  
- **Process:**  
  - Utilized the `data_cleaned_with_corrections.csv` dataset to train an updated Word2Vec model.  
  - Fine-tuned the pretrained `trmodel.bin` for domain-specific vocabulary in hate speech detection.  
  - **Parameters Used:** Customized for optimal performance, including vector size, window, and min count.  
- **Output:** Saved as `updated_trmodel.bin`.

---

## Model Training and Evaluation

- **Pipeline:**  
  - Word embeddings (fine-tuned Word2Vec) were used as input for classification models.  
  - Models applied:  
    - Gradient Boosting (GBM)  
    - XGBoost  
    - Random Forest  
    - CatBoost  
    - Artificial Neural Network (ANN)  

- **Performance Metrics:** Results were evaluated on the `data_cleaned_with_corrections.csv` dataset.  
- **Performance Table:**

| Model          | Precision | Recall | F1-Score | Accuracy |
|----------------|-----------|--------|----------|----------|
| Gradient Boosting | 0.70      | 0.68   | 0.69     | 0.71     |
| XGBoost        | 0.73      | 0.71   | 0.72     | 0.74     |
| Random Forest  | 0.75      | 0.72   | 0.73     | 0.76     |
| CatBoost       | 0.76      | 0.73   | 0.74     | 0.77     |
| ANN            | 0.72      | 0.69   | 0.70     | 0.73     |

---

## Class Imbalance Handling

- **Resampling Methods Applied:**  
  - **Undersampling:** Reduced the majority class size.  
  - **SMOTE:** Generated synthetic samples for the minority class.  
  - **Combination Method:** Applied a hybrid approach using both SMOTE and undersampling.  

- **Impact on Results:**  
  - Improved balance without compromising dataset integrity.  
  - **Performance After Resampling:**

| Model          | Resampling Method | Precision | Recall | F1-Score | Accuracy |
|----------------|-------------------|-----------|--------|----------|----------|
| XGBoost        | SMOTE             | 0.76      | 0.75   | 0.75     | 0.77     |
| Random Forest  | Combination       | 0.78      | 0.76   | 0.77     | 0.79     |

---

## Results and Observations

- **Initial Best Model:** CatBoost with F1-Score of **0.74**.  
- **After Resampling:** Random Forest (F1-Score: **0.77**) showed the best results with a combination method.  
- **Impact of Fine-Tuning:** Fine-tuned Word2Vec embeddings significantly enhanced model performance.  

---

## Future Work

- Experiment with additional transformer models like BERT for text embeddings.  
- Automate hyperparameter tuning with tools like Optuna.  
- Incorporate multilingual datasets for broader application.  

---

### Repository Files

- `data.xlsx`: Raw dataset.  
- `data_cleaned.csv`: Preprocessed dataset.  
- `data_cleaned_with_corrections.csv`: Dataset with corrected tweets.  
- `updated_trmodel.bin`: Fine-tuned Word2Vec model.  
- `final_models/`: Directory containing all trained and fine-tuned models.

---

### Conclusion

This project demonstrates the effectiveness of preprocessing, spelling corrections using Zemberek, and fine-tuned Word2Vec embeddings in hate speech detection on Turkish tweets. Leveraging resampling methods such as SMOTE and hybrid approaches further improved model performance. Future research will explore advanced embeddings and multilingual capabilities for even better outcomes.
