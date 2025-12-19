# bert_base_embed_save.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

DATA_PATH = "/home/detsu/Documents/vscode/NLP/NLP/data/texts_clean.csv"
ART_DIR   = "/home/detsu/Documents/vscode/NLP/NLP/artifacts"

EMB_PATH  = os.path.join(ART_DIR, "bert_base_cls.npy")
Y_PATH    = os.path.join(ART_DIR, "bert_labels.npy")

MODEL_NAME = "bert-base-cased"
MAX_LENGTH = 96        # runtime багасгасан
BATCH_SIZE = 16        # GPU байвал OK, CPU бол 8 болгож болно


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]


def main():
    os.makedirs(ART_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].values.astype(np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    dataset = TextDataset(texts)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_cls = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="BERT CLS embedding"):
            enc = tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :]   # [CLS]
            all_cls.append(cls.cpu())

    X = torch.cat(all_cls, dim=0).numpy()

    np.save(EMB_PATH, X)
    np.save(Y_PATH, labels)

    print("Saved:", EMB_PATH, X.shape)
    print("Saved:", Y_PATH, labels.shape)
    print("Device:", device)


if __name__ == "__main__":
    main()
