from typing import List, Tuple
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from pymorphy2 import MorphAnalyzer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pathlib import Path

nltk.download("stopwords")
nltk.download("punkt")

MODELS_PATH: str = Path(__file__).parent / 'models'

stop_words = set(stopwords.words('russian') + stopwords.words('english'))
morph = MorphAnalyzer()

vect_tfidf: TfidfVectorizer = pickle.load(
    open(MODELS_PATH / "vect_tfidf.p", "rb"))
scaler: MaxAbsScaler = pickle.load(
    open(MODELS_PATH / "scaler.p", "rb"))
log_reg: LogisticRegression = pickle.load(
    open(MODELS_PATH / "log_reg.p", "rb"))


def clean_text(text: str) -> str:
    text = re.sub('[^а-яёА-ЯЁa-zA-z]', ' ', text)
    text = word_tokenize(text.lower())
    text = [morph.normal_forms(token)[0] for token in text
            if token not in stop_words and len(token) > 2]
    text = " ".join(text)
    return text


def get_udcs(text: str) -> List[Tuple[str, str]]:
    text = clean_text(text)
    array = vect_tfidf.transform([text])
    array = scaler.transform(array)
    proba = log_reg.predict_proba(array)
    top_5_classes = np.argsort(proba)[0][::-1][:5]
    udcs = log_reg.classes_[top_5_classes]
    result = [
        (f"{udc}", f"https://teacode.com/online/udc/{udc}/{udc}.html")
        if udc != "00" else ("00", "https://teacode.com/online/udc/00/00.html")
        for udc in udcs
    ]
    return result
