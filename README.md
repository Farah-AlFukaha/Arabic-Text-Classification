# Arabic-Text-Classification

## 📌 Overview
This project focuses on **emotion detection in Arabic text**, a multi-class text classification task.
The goal is to compare **classical machine learning models** and **deep learning approaches**
for identifying emotions expressed in Arabic sentences.

Arabic emotion detection is challenging due to language complexity, dialectal variations,
and limited linguistic resources.

---

## 🎯 Objectives
- Preprocess and normalize Arabic text
- Represent text using multiple vectorization techniques
- Train and evaluate classical machine learning models
- Build and compare deep learning models (FNN and LSTM)
- Analyze and compare all models using standard evaluation metrics

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing
- Text cleaning (punctuation, diacritics, stop words removal)
- Arabic tokenization
- Text normalization and word form handling

### 2️⃣ Text Representation Techniques
- Bag-of-Words (BoW)
- TF-IDF
- Word2Vec (pre-trained embeddings)

---

## 🤖 Models Implemented

### Classical Machine Learning
- Naive Bayes (MultinomialNB)
- Support Vector Machine (RBF Kernel)
- Decision Tree
- Random Forest
- AdaBoost

### Deep Learning
- Feed-Forward Neural Network (FNN)
- LSTM (Recurrent Neural Network)
- *(Optional)* BiLSTM for improved sequence modeling

---

## 📊 Evaluation Metrics
All models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

Comparative analysis is conducted to assess the effectiveness of each representation
and learning approach.

---

## 📂 Dataset
- **EmotionalTone Arabic Dataset**
- Source: EmotionalTone GitHub Repository

---

## 🛠 Tools & Technologies
- Python
- scikit-learn
- TensorFlow / Keras
- NLP libraries for Arabic text processing
- Pre-trained Word2Vec embeddings

---

## 🚀 Results & Insights
- Deep learning models (LSTM) outperform most classical models
- TF-IDF and Word2Vec provide better representations than BoW
- Arabic text preprocessing significantly impacts overall performance

---

## 🌟 Bonus Enhancements
- Transformer-based models (AraBERT / HuggingFace)
- Hyperparameter tuning using GridSearch

---

## 👩‍💻 Author
**Farah Al-Fuqaha**  
Data Science & Artificial Intelligence  
📧 fukaha.farah@gmail.com
🔗 https://www.linkedin.com/in/farah-al-fukaha-628130315
