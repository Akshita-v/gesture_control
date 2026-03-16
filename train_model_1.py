import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load the CSV
df = pd.read_csv('hand_data.csv', header=None)

# 2. Separate Names and Coordinates
y_raw = df.iloc[:, 0]
X = df.iloc[:, 1:]

# 3. Convert Names to Numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)

# --- THE ACCURACY CHECK STEP ---
# This splits your 375 rows into 300 for learning and 75 for testing
X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.2,
	random_state=42,
	stratify=y
)

# 4. Train the Model
print("Training started...")
model = RandomForestClassifier(
	n_estimators=300,
	random_state=42,
	class_weight="balanced_subsample"
)
model.fit(X_train, y_train)

# --- THE RESULTS ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- PERFORMANCE REPORT ---")
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Breakdown per Gesture:")
report_labels = sorted(set(y_test) | set(y_pred))
report_names = [str(encoder.classes_[label]) for label in report_labels]
print(classification_report(y_test, y_pred, labels=report_labels, target_names=report_names, zero_division=0))

# 5. SAVE EVERYTHING (Use the full dataset for the final save)
model.fit(X, y) # Final retrain on all data
joblib.dump(model, 'gesture_model_1.pkl')
joblib.dump(encoder, 'label_encoder_1.pkl')
print("\nFiles saved: gesture_model_1.pkl and label_encoder_1.pkl")