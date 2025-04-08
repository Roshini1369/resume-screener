# save_dummy_model.py

import os
import joblib
from sklearn.linear_model import LogisticRegression

# 1. Create a dummy model
model = LogisticRegression()

# 2. (Optional) Train it on dummy data if needed
# For now, we skip training — just a raw model

# 3. Make sure 'models' folder exists
os.makedirs('models', exist_ok=True)

# 4. Save the model
joblib.dump(model, 'models/model.pkl')

print("✅ Dummy model saved at models/model.pkl")
