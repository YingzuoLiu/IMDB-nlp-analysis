# 1. Data Loading and Initial EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import re
import warnings

warnings.filterwarnings('ignore')

def load_imdb_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def clean_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def basic_text_cleaning(text):
    text = clean_html(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def plot_length_distribution(df, title=""):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='word_count', bins=50)
    plt.title(f'{title} Review Length Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.show()

def analyze_reviews(df, title=""):
    df['word_count'] = df['review'].apply(lambda x: len(str(x).split()))
    plot_length_distribution(df, title)
    return df

train_df, test_df = load_imdb_data('imdb_train.csv', 'imdb_test.csv')
train_df['cleaned_review'] = train_df['review'].apply(basic_text_cleaning)
test_df['cleaned_review'] = test_df['review'].apply(basic_text_cleaning)
train_df = analyze_reviews(train_df, "Training")
test_df = analyze_reviews(test_df, "Test")

# 2. Text Preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def prepare_data_for_modeling(train_reviews, test_reviews=None):
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 1), stop_words='english')
    X_train = tfidf.fit_transform(train_reviews)
    X_test = tfidf.transform(test_reviews) if test_reviews is not None else None
    return X_train, X_test, tfidf

X_train, X_test, vectorizer = prepare_data_for_modeling(train_df['cleaned_review'], test_df['cleaned_review'])

# 3. Baseline Model - Logistic Regression with TF-IDF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

train_labels = np.concatenate([np.ones(50), np.zeros(50)])
test_labels = np.concatenate([np.ones(12500), np.zeros(12500)])

model = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0)
model.fit(X_train, train_labels)
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print("Training Performance:\n", classification_report(train_labels, train_preds))
print("Test Performance:\n", classification_report(test_labels, test_preds))

def plot_confusion_matrix(y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

plot_confusion_matrix(test_labels, test_preds, "Test Set Confusion Matrix")

# 4. Advanced Model - DistilBERT
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, AutoConfig, AutoModel
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index],
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'targets': torch.tensor(self.labels[index], dtype=torch.long)
        }

train_dataset = CustomDataset(train_df['cleaned_review'].tolist(), train_labels, DistilBertTokenizer.from_pretrained('distilbert-base-uncased'), max_len=128)
test_dataset = CustomDataset(test_df['cleaned_review'].tolist(), test_labels, DistilBertTokenizer.from_pretrained('distilbert-base-uncased'), max_len=128)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

class CustomDistilBert(nn.Module):
    def __init__(self, num_classes=2, dropout_prob=0.3):
        super(CustomDistilBert, self).__init__()
        self.distilbert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = nn.Linear(self.distilbert.config.hidden_size, self.distilbert.config.hidden_size)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, attention_mask=None):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.dropout(self.pre_classifier(pooled_output))
        logits = self.classifier(pooled_output)
        return logits

model = CustomDistilBert().to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['targets'].to(device)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch + 1} completed.")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['targets'].to(device)
        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

print("Test Performance:")
print(classification_report(all_labels, all_preds))
