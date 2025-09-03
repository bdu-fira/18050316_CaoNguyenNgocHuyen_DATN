import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize
from flask import Flask, render_template_string, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
from urllib.parse import quote
from flask import Flask, render_template_string, request, redirect, url_for
import webbrowser
from transformers import AutoModel, AutoTokenizer
import os

# Load dữ liệu từ file CSV
df = pd.read_csv("intent_dataset.csv")

# Tiền xử lý: tách từ tiếng Việt
def preprocess(text):
    return word_tokenize(text, format="text").lower()
    
    
def preprocess_text(text):
    # Chuyển thành chữ thường
    text = text.lower()
    # Loại bỏ dấu câu
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Loại bỏ số
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Loại bỏ stopwords
    stop_words = set(stopwords.words('english') + stopwords.words('vietnamese'))
    tokens = [word for word in tokens if word not in stop_words]
    # Ghép lại thành câu
    return ' '.join(tokens)



def bag_of_word(tokenized_sentence, words):
    """
    Chuyển đổi một câu đã được tách từ thành một vector Bag-of-Words.
    Tham số:
    - tokenized_sentence (list): Một danh sách các từ (token) của câu cần xử lý.
                                 Ví dụ: ['chào', 'bạn']
    - words (list): Một danh sách chứa tất cả các từ duy nhất trong bộ từ điển
                      (vocabulary). Danh sách này đã được sắp xếp.
    Kết quả trả về:
    - numpy.ndarray: Một vector Bag-of-Words.
    """
    # Khởi tạo một vector chứa toàn số 0 với độ dài bằng số lượng từ trong từ điển.
    bag = np.zeros(len(words), dtype=np.float32)

    # Duyệt qua từng từ trong câu đầu vào đã được tách từ
    for word_in_sentence in tokenized_sentence:
        # Duyệt qua từng từ trong bộ từ điển
        for i, w in enumerate(words):
            # Nếu từ trong câu khớp với từ trong từ điển
            if w == word_in_sentence:
                # Đánh dấu vị trí tương ứng trong vector `bag` là 1
                bag[i] = 1
                # Không cần tìm tiếp vì đã tìm thấy từ này rồi
                break

    return bag


# Tiền xử lý toàn bộ text
df["text_clean"] = df["text"].apply(preprocess)
#print(df)
# Vector hóa toàn bộ văn bản (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text_clean"])

def detect_intent(query: str):
    """
    Nhận vào câu query, trả về (intent gần nhất, độ tương đồng cao nhất)
    """
    query_clean = preprocess(query)
    query_vec = vectorizer.transform([query_clean])

    # Tính độ tương đồng cosine
    similarities = cosine_similarity(query_vec, X)[0]  # lấy dòng đầu (vì chỉ 1 query)

    # Tìm index có độ tương đồng cao nhất
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    best_intent = df.iloc[best_idx]["intent"]

    return best_intent, float(best_score)


def predict_intent(query, top_n=3):
    # Tiền xử lý query
    processed_query = preprocess_text(query)
    # Chuyển đổi thành vector
    query_vector = tfidf_vectorizer.transform([processed_query])
    
    # Tính toán độ tương đồng với tất cả các intent
    similarities = {}
    for intent, embedding in avg_intent_embeddings.items():
        # Reshape embedding để đảm bảo cùng kích thước
        embedding_reshaped = embedding.reshape(1, -1)
        similarity = cosine_similarity(query_vector, embedding_reshaped)[0][0]
        similarities[intent] = similarity
    
    # Sắp xếp theo độ tương đồng giảm dần
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # Lấy top_n intents
    top_intents = []
    for intent, similarity in sorted_similarities[:top_n]:
        intent_name = label_encoder.inverse_transform([intent])[0]
        top_intents.append((intent_name, similarity))
    
    return top_intents



