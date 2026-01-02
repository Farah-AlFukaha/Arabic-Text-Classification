
import re
import emoji
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Arabic stopwords
stop_words = set(stopwords.words('arabic'))


def is_clean_arabic_word(word):
    return re.fullmatch(r'[ء-ي]{3,}', word) is not None

def remove_diacritics(text):
    diacritics_pattern = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    return re.sub(diacritics_pattern, '', text)


def preprocess_arabic_text(text):
    if not isinstance(text, str):
        return ""
    
    
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[A-Za-z0-9٠-٩]", "", text)
    text = re.sub(r'(.)\1+', r'\1', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = emoji.demojize(text, language='ar')
    
    text = re.sub(r"[ؤۆ]", "و", text)
    text = re.sub(r"[ء]", "", text)
    text = re.sub(r"[گڪ]", "ك", text)
    text = re.sub(r"[پچژڤ]", "", text)
    text = re.sub(r"[ۀ]", "ه", text)
    text = re.sub("ى", "ي", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = remove_diacritics(text)
    text = re.sub(r"([!?؟])", r" \1 ", text)
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    words = word_tokenize(text)
    cleaned_words = [w for w in words if is_clean_arabic_word(w) and w not in stop_words]
    
    return " ".join(cleaned_words)
