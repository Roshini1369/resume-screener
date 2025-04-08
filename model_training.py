import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('your_cv_dataset.csv')  # Modify with your actual dataset

# Assuming your CSV has 'text' and 'label' columns
texts = df['text'].tolist()
labels = df['label'].tolist()

# Generate BERT embeddings
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = bert_model.encode(texts)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Train a simple classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'models/model.pkl')

print("✅ Model trained and saved!")
