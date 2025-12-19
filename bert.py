import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __getitem__(self, idx):
        return self.texts[idx]

    def __len__(self):
        return len(self.texts)


def bert_tokenizer(tokenizer, batch_texts, max_length=128):
    token = {'input_ids': [], 'attention_mask': []}

    for sent in batch_texts:
        encoded = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True
        )
        token["input_ids"].append(torch.tensor(encoded["input_ids"]))
        token["attention_mask"].append(torch.tensor(encoded["attention_mask"]))

    token["input_ids"] = torch.stack(token["input_ids"], dim=0)
    token["attention_mask"] = torch.stack(token["attention_mask"], dim=0)
    return token

def word_embed(device, tokenizer, bert, column, max_length=128, batch_size=8):

    dataset = TextDataset(column)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    out_list = []

    for batch_texts in tqdm(dataloader, desc="BERT embedding"):
        batch_tokens = bert_tokenizer(tokenizer, batch_texts, max_length)

        for key in batch_tokens:
            batch_tokens[key] = batch_tokens[key].to(device)

        with torch.no_grad():
            outputs = bert(**batch_tokens)
          
            cls_embed = outputs.last_hidden_state[:, 0, :]

        out_list.append(cls_embed.cpu())

    final = torch.cat(out_list, dim=0)
    return final



if __name__ == "__main__":

    df = pd.read_csv("/home/detsu/Documents/vscode/NLP/NLP/data/texts_clean.csv")
    texts = df["text"].tolist()
    labels = df["label"].values

  
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert.to(device)
    bert.eval()

    
    embeddings = word_embed(
        device=device,
        tokenizer=tokenizer,
        bert=bert,
        column=texts,
        max_length=128,
        batch_size=8
    )

    print("Embedding shape:", embeddings.shape)
    embeddings_np = embeddings.numpy()
    
  
    import os
    os.makedirs("artifacts", exist_ok=True)

    np.save("artifacts/bert_base_cls.npy", embeddings_np)
    np.save("artifacts/bert_labels.npy", labels)

    print("Saved:")
    print("artifacts/bert_base_cls.npy")
    print("artifacts/bert_labels.npy")