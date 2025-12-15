Sentiment Analysis with TF-IDF, Word2Vec, GloVe & RNN Models

This notebook performs large-scale sentiment analysis on the **1.6M Tweets dataset**, evaluating different embedding methods and RNN architectures.

---

## 📌 Data Preprocessing
- Remove URLs, mentions, punctuation  
- Convert emojis → text  
- Tokenize using TweetTokenizer + WordPiece tokenizer  
- Create cleaned subset of 200k tweets  
- Train/Val/Test split: 180k / 9k / 9k  

---

## 📌 Embedding Methods Implemented
### **1. TF-IDF**
Classical sparse vector representation  
Used with fully connected neural network classifier  

### **2. Word2Vec**
Trained on the cleaned tweet corpus  
Used to build embedding matrix for RNN models  

### **3. GloVe (Twitter 200d)**
Loaded pretrained embeddings  
Also used with RNN models  

---

## 📌 RNN Architectures Evaluated
- **LSTM**  
- **GRU**  
- **Bi-LSTM**  

All models include:
- Embedding layer  
- Dropout  
- Dense sigmoid output  

Metrics reported:
- Accuracy  
- Precision  
- Recall  
- F1 score  
- Training & validation curves  

---

## 📊 Key Results
- TF-IDF models achieve ~78–79% accuracy  
- RNN models with embeddings achieve ~78–80%  
- Precision/Recall vary with embedding type  

---

## 🛠️ Libraries
TensorFlow/Keras • scikit-learn • NLTK • Gensim Word2Vec • Transformers • Matplotlib

