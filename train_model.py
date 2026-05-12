import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Dataset
data = pd.DataFrame({

    'fever': [1,1,1,0,0,1,0,1],
    'cough': [1,0,1,1,0,0,1,1],
    'headache': [1,1,0,0,0,1,0,1],
    'bodypain': [1,1,1,0,0,1,0,1],

    'disease': [
        'Flu',
        'Migraine',
        'COVID-19',
        'Cold',
        'Healthy',
        'Dengue',
        'Allergy',
        'Typhoid'
    ]
})

# Features
X = data[['fever', 'cough', 'headache', 'bodypain']]

# Labels
y = data['disease']

# Train model
model = RandomForestClassifier()

model.fit(X, y)

# Save model
pickle.dump(model, open('disease_model.pkl', 'wb'))

print("Model trained successfully!")