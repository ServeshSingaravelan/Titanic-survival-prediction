import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

data_dir = "/builds/ainb_24_semper_fortis/titanic_model_service/"
train_data = pd.read_csv(os.path.join(data_dir, "w_train.csv"))
test_data = pd.read_csv(os.path.join(data_dir, "w_test.csv"))
model_dir = "/builds/ainb_24_semper_fortis/titanic_model_service/models/"


def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def preprocess_data(data):
    imputer_age = SimpleImputer(strategy="median")
    data['Age'] = imputer_age.fit_transform(data[['Age']]).ravel()
    imputer_fare = SimpleImputer(strategy="median")
    data['Fare'] = imputer_fare.fit_transform(data[['Fare']]).ravel()
    imputer_embarked = SimpleImputer(strategy="most_frequent")
    data['Embarked'] = imputer_embarked.fit_transform(data[['Embarked']]).ravel()
    data['TraveledAlone'] = ((data['SibSp'] + data['Parch']) == 0).astype(int)
    encoder_sex = LabelEncoder()
    data['Sex'] = encoder_sex.fit_transform(data['Sex'])
    encoder_embarked = LabelEncoder()
    data['Embarked'] = encoder_embarked.fit_transform(data['Embarked'])
    scaler = StandardScaler()
    data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])
    return data, scaler

train_data, scaler = preprocess_data(train_data)

X = train_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'TraveledAlone']]
y = train_data['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'knn': KNeighborsClassifier(),
    'random_forest': RandomForestClassifier(random_state=42),
    'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
}
params = {
    'knn': {'n_neighbors': range(3, 16, 2)},
    'random_forest': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None]},
    'logistic_regression': {'C': [0.1, 1, 10, 100]}
}

for name, model in models.items():
    model_path = model_dir + f'{name}_model.pkl'
    if os.path.exists(model_path):
        models[name] = load_model(model_path)
        print(f"Loaded {name} model from file.")
    else:
        grid = GridSearchCV(model, params[name], cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        models[name] = grid.best_estimator_
        with open(model_path, 'wb') as file:
            pickle.dump(models[name], file)
            print(f"{models[name]} Model saved at {model_path}")

def predict_survival(Pclass, Sex, Age, Fare, TraveledAlone, Embarked, model_choice, model_choice2, model_choice3):
    input_data = pd.DataFrame({
        'Pclass': [Pclass],
        'Sex': [LabelEncoder().fit_transform([Sex])[0]],
        'Age': [Age],
        'Fare': [Fare],
        'Embarked': [LabelEncoder().fit_transform([Embarked])[0]],
        'TraveledAlone': [1 if TraveledAlone.lower() == 'yes' else 0]
    })
    
    input_data[['Age', 'Fare']] = scaler.transform(input_data[['Age', 'Fare']])
    
    results = []
    
    for model_name in [model_choice, model_choice2, model_choice3]:
        if model_name != "" :
            model_path = model_dir + f'{model_name}_model.pkl'
            model = load_model(model_path)
            probability = model.predict_proba(input_data)[0][1]
            results.append(f"The predicted probability of survival with {model_name} model is {probability*100:.2f}%" )
    
    return results

#reference for me
# print(predict_survival(3, 'male', 80, 8, 'no', 'Southampton', 'knn','random_forest','logistic_regression'))