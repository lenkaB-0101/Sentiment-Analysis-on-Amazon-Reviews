from sklearn.metrics import classification_report

# Vyhodnocení modelu
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, labels = batch
        outputs = model(input_ids)
        _, predicted = torch.max(outputs.logits, dim=1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

# Výsledky metrik
print(classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"]))
