### Detection of Hate Speech on Turkish Tweets Using Machine Learning With Fine-Tuning

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

- **Tool Used:** Zemberek Library ( [Reference](https://github.com/ahmetaa/zemberek-nlp/tree/master/normalization)).  
- **Process:**  
  - Morphological analysis using `TurkishMorphology`.  
  - Spelling corrections via `TurkishSpellChecker`.  
- **Result:** Added a `corrected_tweet` column to the dataset.  
- **Output:** The corrected data was saved as `data_cleaned_with_corrections.csv`.

![image](https://github.com/user-attachments/assets/01dd2951-f682-43a6-b0c5-3681913efadc)

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

---
![image](https://github.com/user-attachments/assets/e47b0bc3-c5c2-4ff4-88ba-5dabd8d190b1)

## Class Imbalance Handling

- **Resampling Methods Applied:**  
  - **Undersampling:** Reduced the majority class size.  
  - **SMOTE:** Generated synthetic samples for the minority class.  
  - **Combination Method:** Applied a hybrid approach using both SMOTE and undersampling.  

- **Impact on Results:**  
  - Improved balance without compromising dataset integrity.  
  - **Performance After Resampling:**
**After Undersampling Process**
![image](https://github.com/user-attachments/assets/be140421-ab78-46f0-8da3-12e48729160e)

**After Oversampling(SMOTE) Process**
![image](https://github.com/user-attachments/assets/b4516d5b-c6f9-4c03-85e8-a7c224b778f1)


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
