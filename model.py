import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Training Data: (Pitch, Speed, Emotion Score)
X_train = np.array([
    [200, 90, -0.5],  
    [180, 120, 0.3],  
    [250, 85, -0.8],  
])
y_train = np.array([1, 0, 1])  # 1 = Lie, 0 = Truth

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

def predict_lie(features):
    prediction = model.predict([features])
    return "LIE" if prediction[0] == 1 else "TRUTH"

if __name__ == "__main__":
    sample_features = [210, 88, -0.6]
    result = predict_lie(sample_features)
    print("🕵️‍♂️ AI Verdict:", result)