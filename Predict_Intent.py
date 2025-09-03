import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ===== 1. Khôi phục mô hình và tokenizer =====
model_path = "bert_intent_model"  # Thư mục chứa mô hình đã lưu

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ===== 2. Gắn lại LabelEncoder từ dữ liệu gốc =====
# Vì LabelEncoder không lưu cùng mô hình, ta cần khôi phục lại từ file CSV ban đầu
df = pd.read_csv("intent_dataset.csv")
df['intent'] = df['intent'].astype(str).str.strip()

label_encoder = LabelEncoder()
label_encoder.fit(df['intent'])

# ===== 3. Hàm dự đoán ý định người dùng =====
def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([pred])[0]

# ===== 4. Dùng thử =====
while True:
    query = input("Nhập câu truy vấn người dùng (hoặc 'exit'): ").strip()
    if query.lower() == "exit":
        break
    intent = predict_intent(query)
    print(f"-> Ý định dự đoán: {intent}\n")
