# app.py
from flask import Flask, render_template, request, jsonify
from urllib.parse import quote_plus
import webbrowser      # chỉ dùng khi chạy local (server side)
from intent import detect_intent
from flask import Flask, render_template_string, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
from urllib.parse import quote
from flask import Flask, render_template_string, request, redirect, url_for
import webbrowser
from transformers import AutoModel, AutoTokenizer
import os



app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/intent", methods=["POST"])
def intent_api():
    data = request.get_json(force=True)
    query = data.get("query", "")
    label, score = detect_intent(query)
    #return jsonify({"intent": label, "score": score})
    #print({"Điểm số": score})
    #print("Nhãn:", label)
    label = label.replace("_"," ")
    label = label.capitalize()
    tieude = "Ý định người dùng muốn truy vấn:"
    label = f"{tieude} {label}"
    return jsonify({"intent": label})


@app.route("/smart-search", methods=["POST"])
def smart_search():
    data = request.get_json(force=True)
    query  = data.get("query", "")
    intent = data.get("intent", "")
    #print(intent)
    intent = intent.split(" (")[0]
    intent = intent.replace("Ý định người dùng muốn truy vấn:","")
    full_q = f"{query} : {intent}"
    url    = f"https://www.google.com/search?q={quote_plus(full_q)}"

    # Mở tab mới trên *máy local* chạy Flask – thuận tiện khi bạn demo offline.
    webbrowser.open_new_tab(url)

    # Trả về URL để JavaScript cũng có thể mở ở trình duyệt client.
    return jsonify({"redirect": url})

if __name__ == "__main__":
    app.run(debug=True)
