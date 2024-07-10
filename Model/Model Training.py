import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Read Gesture Labels
gestures = pd.read_csv('gestures.csv', header=None)[0].tolist()

# Step 2: Combine Data from Text Files
data_frames = []
for gesture in gestures:
    gesture_data = pd.read_csv(f'{gesture}_data.txt', header=None)
    gesture_data['label'] = gesture
    data_frames.append(gesture_data)

combined_data = pd.concat(data_frames, ignore_index=True)

# Ensure feature columns are numerical
for col in combined_data.columns[:-1]:
    combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce')

# Drop rows with non-numeric values in feature columns
combined_data = combined_data.dropna()

combined_data.to_csv('combined_gesture_data.csv', index=False)

# Step 3: Data Preprocessing
data = pd.read_csv('combined_gesture_data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, 'gesture_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
