import pytest
import os
import pickle
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from app.predictornew import load_model, preprocess_data, predict_survival

# Test loading model
def test_load_model(tmpdir):
    model_path = os.path.join(tmpdir, 'test_model.pkl')
    model = KNeighborsClassifier()
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    loaded_model = load_model(model_path)
    assert isinstance(loaded_model, KNeighborsClassifier)

# Test loading model with non-existent file
def test_load_model_non_existent():
    with pytest.raises(FileNotFoundError):
        load_model('non_existent_model.pkl')

# Test preprocess_data function
def test_preprocess_data():
    data = pd.DataFrame({
        'Pclass': [3, 1],
        'Sex': ['male', 'female'],
        'Age': [22, 38],
        'Fare': [7.25, 71.2833],
        'Embarked': ['S', 'C'],
        'SibSp': [1, 1],
        'Parch': [0, 0]
    })
    processed_data, scaler = preprocess_data(data)
    assert 'TraveledAlone' in processed_data.columns
    assert 'Sex' in processed_data.columns
    assert 'Embarked' in processed_data.columns
    assert processed_data['Age'].notnull().all()
    assert processed_data['Fare'].notnull().all()
    assert processed_data['Embarked'].notnull().all()

# Test predict_survival function
def test_predict_survival():
    result = predict_survival(3, 'male', 22, 7.25, 'no', 'S', 'random_forest', '', '')
    assert isinstance(result, list)
    assert "predicted probability of survival" in result[0]

# Test predict_survival with invalid model name
def test_predict_survival_invalid_model():
    with pytest.raises(FileNotFoundError):
        predict_survival(3, 'male', 22, 7.25, 'no', 'S', 'invalid_model', '', '')

# Test predict_survival with edge case ages
def test_predict_survival_edge_case_ages():
    result_newborn = predict_survival(3, 'male', 0.5, 7.25, 'no', 'S', 'random_forest', '', '')
    result_elderly = predict_survival(1, 'female', 100, 50.0, 'yes', 'C', 'random_forest', '', '')
    assert isinstance(result_newborn, list)
    assert isinstance(result_elderly, list)
    assert "predicted probability of survival" in result_newborn[0]
    assert "predicted probability of survival" in result_elderly[0]

# Test predict_survival with all model choices
def test_predict_survival_all_models():
    result = predict_survival(3, 'male', 22, 7.25, 'no', 'S', 'random_forest', 'knn', 'logistic_regression')
    assert isinstance(result, list)
    assert len(result) == 3
    for res in result:
        assert "predicted probability of survival" in res
