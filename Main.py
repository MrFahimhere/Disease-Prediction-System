import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
print(final_df.head(30))
# print(final_df.info()) checking if the column is integer

#modelling
# x=final_df.drop('Disease_new',axis=1)
# y=final_df['Disease_new']
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# model=RandomForestClassifier(n_estimators=100,random_state=42)
# model.fit(x_train,y_train)
# y_predict=model.predict(x_test)
# accuracy=accuracy_score(y_test,y_predict)
# print("The accuracy score is ",accuracy)


