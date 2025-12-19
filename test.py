import pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset, DataLoader, random_split


model_name = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


df = pd.read_csv("/home/detsu/Documents/vscode/NLP/NLP/data/texts_clean.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()


def bert_tokenizer(tokenizer, data, max_length):
    token = {'input_ids': [], 'attention_mask': []}
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True
        )
        token['input_ids'].append(torch.tensor(encoded_sent['input_ids']))
        token['attention_mask'].append(torch.tensor(encoded_sent['attention_mask']))
    token['input_ids'] = torch.stack(token['input_ids'])
    token['attention_mask'] = torch.stack(token['attention_mask'])
    return token

max_length = 128
tokens = bert_tokenizer(tokenizer, texts, max_length)


input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]
labels = torch.tensor(labels)

dataset = TensorDataset(input_ids, attention_mask, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)



import torch
import numpy as np
from tqdm import tqdm

model.eval()
all_embeddings = []

with torch.no_grad():
    for i in tqdm(range(0, len(tokens['input_ids']), 16)):   # batch = 16
        batch_ids = tokens["input_ids"][i:i+16].to(device)
        batch_mask = tokens["attention_mask"][i:i+16].to(device)

        outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
        cls_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()   # CLS токен
        all_embeddings.append(cls_embeds)

final_embeddings = np.vstack(all_embeddings)

np.save("artifacts/bert_cls_embeddings.npy", final_embeddings)
print("BERT CLS embeddings saved:", final_embeddings.shape)
