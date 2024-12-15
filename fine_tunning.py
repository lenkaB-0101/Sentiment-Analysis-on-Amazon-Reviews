from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
import torch

# Máme dataset (texty a sentimenty)
reviews = ["Perfect%", "Great%", "Not what I expected"]
sentiments = [2, 0, 1]  # 0: Negative, 1: Neutral, 2: Positive

# Načteme tokenizer a model DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Tokenizace textů
inputs = tokenizer(reviews, padding=True, truncation=True, return_tensors='pt')

# Rozdělení na trénovací a testovací data
X_train, X_test, y_train, y_test = train_test_split(inputs['input_ids'], sentiments, test_size=0.2)

# Vytvoření DataLoaderu
train_data = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_data = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8)

# Optimalizátor
optimizer = AdamW(model.parameters(), lr=2e-5)

# Trénování modelu
model.train()
for epoch in range(3):
    for batch in train_loader:
        input_ids, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, labels=torch.tensor(labels))
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item()}")

# Vyhodnocení modelu
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids, labels = batch
        outputs = model(input_ids)
        _, predicted = torch.max(outputs.logits, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")
