import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

# preprocessing 
df=pd.read_csv("Disease_dataset.csv")
df = df.dropna(axis=1, how='all')
symptom_cols = [col for col in df.columns if "Symptom" in col]
df['all_symptoms'] = df[symptom_cols].values.tolist()
df['all_symptoms'] = df['all_symptoms'].apply(lambda x: [i for i in x if isinstance(i, str)])
mlb = MultiLabelBinarizer()
symptom_encoded = mlb.fit_transform(df['all_symptoms'])
symptom_df = pd.DataFrame(symptom_encoded, columns=mlb.classes_)
final_df = pd.concat([df['Disease'], symptom_df], axis=1)
lb=LabelEncoder()
final_df['Disease_new']=lb.fit_transform(df.Disease)
final_df.drop('Disease',axis=1,inplace=True)
print(final_df.head())
# print(final_df.info()) checking if the column is integer

#modelling
x=final_df.drop('Disease_new',axis=1)
y=final_df['Disease_new']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
accuracy=accuracy_score(y_test,y_predict)
print("The accuracy score is ",accuracy)

# Save model artifacts for Streamlit app
joblib.dump(model, "model.pkl")
joblib.dump(mlb, "mlb.pkl")
joblib.dump(lb, "lb.pkl")
joblib.dump(accuracy, "accuracy.pkl")
joblib.dump(list(mlb.classes_), "symptoms.pkl")

print("Model artifacts saved: model.pkl, mlb.pkl, lb.pkl, accuracy.pkl, symptoms.pkl")
# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_predict)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    cmap="Blues",
    linewidths=0.3,
    linecolor='white',
    square=True,
    cbar=True
)

plt.title("Confusion Matrix", fontsize=14, pad=12)
plt.xlabel("Predicted Label", fontsize=11)
plt.ylabel("True Label", fontsize=11)
plt.xticks(fontsize=7, rotation=90)
plt.yticks(fontsize=7, rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()