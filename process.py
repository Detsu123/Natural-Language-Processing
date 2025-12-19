
import os, re, tarfile, argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse





# ========== regex helper ==========
HTML_TAG_RE = re.compile(r'<[^>]+>')
NON_ALPHA_RE = re.compile(r'[^a-zA-Z]')

# ========== util функцууд ==========
def load_imdb(dataset_path: str):
    texts, labels = [], []
    path = Path(dataset_path)
    if path.is_file() and str(path).endswith(('.tar.gz', '.tgz')):
        extract_dir = path.parent / "_extracted_aclImdb"
        extract_dir.mkdir(exist_ok=True, parents=True)
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        root = extract_dir / "aclImdb"
    elif path.is_dir():
        root = path
    else:
        raise FileNotFoundError("Dataset not found.")

    for split in ["train", "test"]:
        for label, val in [("pos", 1), ("neg", 0)]:
            folder = root / split / label
            for f in folder.glob("*.txt"):
                texts.append(f.read_text(encoding="utf-8", errors="ignore"))
                labels.append(val)
    return texts, np.array(labels, dtype=np.int32)

def clean_text(text: str):
    text = text.lower()
    text = HTML_TAG_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--ngram_max", type=int, default=2)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    (out_root/"data").mkdir(exist_ok=True)
    (out_root/"artifacts").mkdir(exist_ok=True)

    # NLTK data
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

    # spaCy
    try:
        import spacy
        nlp = spacy.load(args.spacy_model, disable=["parser","ner","textcat"])
        use_spacy = True
    except Exception:
        nlp = None
        use_spacy = False

    print("[INFO] Loading IMDB dataset...")
    texts_raw, labels = load_imdb(args.dataset_path)

    print("[INFO] Cleaning text...")
    cleaned = [clean_text(t) for t in texts_raw]
    tokenized = [[w for w in word_tokenize(t)
                  if w.isalpha() and w not in stop_words] for t in cleaned]

    if use_spacy:
        print("[INFO] Lemmatizing with spaCy...")
        texts = []
        for toks in tokenized:
            doc = nlp(" ".join(toks))
            lemmas = [t.lemma_ for t in doc if t.lemma_.isalpha()]
            texts.append(" ".join(lemmas))
    else:
        texts = [" ".join(toks) for toks in tokenized]

    df = pd.DataFrame({"text": texts, "label": labels})
    df.to_csv(out_root/"data"/"texts_clean.csv", index=False)
    print("[OK] Saved cleaned CSV.")

    # -------- TF-IDF vectorization --------
    print("[INFO] Building TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_range=(1, args.ngram_max),
        sublinear_tf=True,
        norm="l2"
    )
    X = vectorizer.fit_transform(df["text"])
    y = labels

    sparse.save_npz(out_root/"artifacts"/"features.npz", X)
    np.save(out_root/"artifacts"/"labels.npy", y) 
    with open(out_root/"artifacts"/"vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    meta = dict(
        num_samples=int(X.shape[0]),
        num_features=int(X.shape[1]),
        used_spacy=use_spacy,
        vectorizer="TF-IDF"
    )  
    json.dump(meta, open(out_root/"artifacts"/"meta.json","w"), indent=2)
    print("[DONE] TF-IDF features saved!")

if __name__ == "__main__":
    main()
