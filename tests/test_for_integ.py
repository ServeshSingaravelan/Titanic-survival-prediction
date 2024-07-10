from fastapi.testclient import TestClient
from app.task import app

client = TestClient(app)

# Test the /survivalchance endpoint with valid input
def test_survivalchance_valid():
    response = client.post("/survivalchance", json={
        "user_pclass": 1,
        "user_sex": "female",
        "user_fare": 100.0,
        "user_age": 29,
        "user_travelled_alone": "yes",
        "user_embarked": "C",
        "user_prediction_model": "random_forest",
        "user_prediction_model_2": "",
        "user_prediction_model_3": ""
    })
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 1
    assert "predicted probability of survival" in response.json()[0]

# Test the /survivalchance endpoint with multiple models
def test_survivalchance_multiple_models():
    response = client.post("/survivalchance", json={
        "user_pclass": 3,
        "user_sex": "male",
        "user_fare": 7.25,
        "user_age": 22,
        "user_travelled_alone": "no",
        "user_embarked": "S",
        "user_prediction_model": "random_forest",
        "user_prediction_model_2": "knn",
        "user_prediction_model_3": "logistic_regression"
    })
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 3
    for res in response.json():
        assert "predicted probability of survival" in res

# Test the /survivalchance endpoint with missing fields
def test_survivalchance_missing_fields():
    response = client.post("/survivalchance", json={
        "user_pclass": 1,
        "user_sex": "female",
        "user_fare": 100.0,
        "user_age": 29,
        "user_travelled_alone": "yes",
        "user_embarked": "C"
    })
    assert response.status_code == 422  # Unprocessable Entity due to missing fields

# Test the /survivalchance endpoint with edge case values
def test_survivalchance_edge_cases():
    response = client.post("/survivalchance", json={
        "user_pclass": 3,
        "user_sex": "male",
        "user_fare": 0.0,
        "user_age": 0.5,
        "user_travelled_alone": "yes",
        "user_embarked": "S",
        "user_prediction_model": "random_forest",
        "user_prediction_model_2": "knn",
        "user_prediction_model_3": "logistic_regression"
    })
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 3
    for res in response.json():
        assert "predicted probability of survival" in res

    response = client.post("/survivalchance", json={
        "user_pclass": 1,
        "user_sex": "female",
        "user_fare": 512.3292,
        "user_age": 80,
        "user_travelled_alone": "no",
        "user_embarked": "C",
        "user_prediction_model": "random_forest",
        "user_prediction_model_2": "",
        "user_prediction_model_3": ""
    })
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert "predicted probability of survival" in response.json()[0]

# Test the /survivalchance endpoint with combinations of different Pclass, Sex, and Embarked
def test_survivalchance_various_combinations():
    combinations = [
        {"user_pclass": 1, "user_sex": "female", "user_embarked": "C"},
        {"user_pclass": 1, "user_sex": "male", "user_embarked": "C"},
        {"user_pclass": 2, "user_sex": "female", "user_embarked": "Q"},
        {"user_pclass": 2, "user_sex": "male", "user_embarked": "Q"},
        {"user_pclass": 3, "user_sex": "female", "user_embarked": "S"},
        {"user_pclass": 3, "user_sex": "male", "user_embarked": "S"},
    ]
    
    for combo in combinations:
        response = client.post("/survivalchance", json={
            "user_pclass": combo["user_pclass"],
            "user_sex": combo["user_sex"],
            "user_fare": 50.0,
            "user_age": 30,
            "user_travelled_alone": "yes",
            "user_embarked": combo["user_embarked"],
            "user_prediction_model": "random_forest",
            "user_prediction_model_2": "knn",
            "user_prediction_model_3": "logistic_regression"
        })
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) == 3
        for res in response.json():
            assert "predicted probability of survival" in res

# Test the /survivalchance endpoint with different travel statuses
def test_survivalchance_travel_status():
    response = client.post("/survivalchance", json={
        "user_pclass": 1,
        "user_sex": "female",
        "user_fare": 100.0,
        "user_age": 29,
        "user_travelled_alone": "no",
        "user_embarked": "C",
        "user_prediction_model": "random_forest",
        "user_prediction_model_2": "knn",
        "user_prediction_model_3": "logistic_regression"
    })
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 3
    for res in response.json():
        assert "predicted probability of survival" in res

    response = client.post("/survivalchance", json={
        "user_pclass": 1,
        "user_sex": "female",
        "user_fare": 100.0,
        "user_age": 29,
        "user_travelled_alone": "yes",
        "user_embarked": "C",
        "user_prediction_model": "random_forest",
        "user_prediction_model_2": "knn",
        "user_prediction_model_3": "logistic_regression"
    })
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 3
    for res in response.json():
        assert "predicted probability of survival" in res

# Test the /survivalchance endpoint with extreme fares
def test_survivalchance_extreme_fares():
    response = client.post("/survivalchance", json={
        "user_pclass": 1,
        "user_sex": "female",
        "user_fare": 512.3292,
        "user_age": 29,
        "user_travelled_alone": "yes",
        "user_embarked": "C",
        "user_prediction_model": "random_forest",
        "user_prediction_model_2": "knn",
        "user_prediction_model_3": "logistic_regression"
    })
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 3
    for res in response.json():
        assert "predicted probability of survival" in res

    response = client.post("/survivalchance", json={
        "user_pclass": 3,
        "user_sex": "male",
        "user_fare": 0.0,
        "user_age": 29,
        "user_travelled_alone": "yes",
        "user_embarked": "S",
        "user_prediction_model": "random_forest",
        "user_prediction_model_2": "knn",
        "user_prediction_model_3": "logistic_regression"
    })
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 3
    for res in response.json():
        assert "predicted probability of survival" in res

# Test the /survivalchance endpoint with boundary age values
def test_survivalchance_boundary_ages():
    response = client.post("/survivalchance", json={
        "user_pclass": 2,
        "user_sex": "male",
        "user_fare": 30.0,
        "user_age": 0.0,  # Newborn
        "user_travelled_alone": "yes",
        "user_embarked": "S",
        "user_prediction_model": "random_forest",
        "user_prediction_model_2": "knn",
        "user_prediction_model_3": "logistic_regression"
    })
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 3
    for res in response.json():
        assert "predicted probability of survival" in res
        response = client.post("/survivalchance", json={
        "user_pclass": 2,
        "user_sex": "female",
        "user_fare": 30.0,
        "user_age": 100.0,  # Very old age
        "user_travelled_alone": "no",
        "user_embarked": "Q",
        "user_prediction_model": "random_forest",
        "user_prediction_model_2": "knn",
        "user_prediction_model_3": "logistic_regression"
    })
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 3
    for res in response.json():
        assert "predicted probability of survival" in res

# Test the /survivalchance endpoint with various age groups
def test_survivalchance_various_ages():
    ages = [1, 5, 12, 18, 25, 35, 45, 55, 65, 75, 85]
    for age in ages:
        response = client.post("/survivalchance", json={
            "user_pclass": 2,
            "user_sex": "female",
            "user_fare": 50.0,
            "user_age": age,
            "user_travelled_alone": "no",
            "user_embarked": "Q",
            "user_prediction_model": "random_forest",
            "user_prediction_model_2": "knn",
            "user_prediction_model_3": "logistic_regression"
        })
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) == 3
        for res in response.json():
            assert "predicted probability of survival" in res

# Test the /survivalchance endpoint with various fare amounts
def test_survivalchance_various_fares():
    fares = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0, 500.0]
    for fare in fares:
        response = client.post("/survivalchance", json={
            "user_pclass": 3,
            "user_sex": "male",
            "user_fare": fare,
            "user_age": 30,
            "user_travelled_alone": "yes",
            "user_embarked": "S",
            "user_prediction_model": "random_forest",
            "user_prediction_model_2": "knn",
            "user_prediction_model_3": "logistic_regression"
        })
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) == 3
        for res in response.json():
            assert "predicted probability of survival" in res

# Test the /survivalchance endpoint with different classes and survival chances
def test_survivalchance_different_classes():
    classes = [1, 2, 3]
    for pclass in classes:
        response = client.post("/survivalchance", json={
            "user_pclass": pclass,
            "user_sex": "female",
            "user_fare": 50.0,
            "user_age": 30,
            "user_travelled_alone": "no",
            "user_embarked": "C",
            "user_prediction_model": "random_forest",
            "user_prediction_model_2": "knn",
            "user_prediction_model_3": "logistic_regression"
        })
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) == 3
        for res in response.json():
            assert "predicted probability of survival" in res
# Test the /survivalchance endpoint with different travel statuses
def test_survivalchance_different_travel_statuses():
    travel_statuses = ["yes", "no"]
    for travel_status in travel_statuses:
        response = client.post("/survivalchance", json={
            "user_pclass": 2,
            "user_sex": "male",
            "user_fare": 75.0,
            "user_age": 40,
            "user_travelled_alone": travel_status,
            "user_embarked": "Q",
            "user_prediction_model": "random_forest",
            "user_prediction_model_2": "knn",
            "user_prediction_model_3": "logistic_regression"
        })
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) == 3
        for res in response.json():
            assert "predicted probability of survival" in res

# Test the /survivalchance endpoint with different sexes and embarkation points
def test_survivalchance_sex_and_embarkation():
    sexes = ["male", "female"]
    embarkation_points = ["C", "Q", "S"]
    for sex in sexes:
        for embark in embarkation_points:
            response = client.post("/survivalchance", json={
                "user_pclass": 3,
                "user_sex": sex,
                "user_fare": 25.0,
                "user_age": 50,
                "user_travelled_alone": "yes",
                "user_embarked": embark,
                "user_prediction_model": "random_forest",
                "user_prediction_model_2": "knn",
                "user_prediction_model_3": "logistic_regression"
            })
            assert response.status_code == 200
            assert isinstance(response.json(), list)
            assert len(response.json()) == 3
            for res in response.json():
                assert "predicted probability of survival" in res

# Test the /survivalchance endpoint with very high and very low fare amounts
def test_survivalchance_extreme_fare_values():
    extreme_fares = [0.01, 1000.0]
    for fare in extreme_fares:
        response = client.post("/survivalchance", json={
            "user_pclass": 1,
            "user_sex": "female",
            "user_fare": fare,
            "user_age": 30,
            "user_travelled_alone": "no",
            "user_embarked": "C",
            "user_prediction_model": "random_forest",
            "user_prediction_model_2": "knn",
            "user_prediction_model_3": "logistic_regression"
        })
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) == 3
        for res in response.json():
            assert "predicted probability of survival" in res
# Test the /survivalchance endpoint with combinations of all parameters
def test_survivalchance_all_combinations():
    combinations = [
        {"user_pclass": 1, "user_sex": "female", "user_fare": 100.0, "user_age": 25, "user_travelled_alone": "yes", "user_embarked": "C"},
        {"user_pclass": 1, "user_sex": "male", "user_fare": 50.0, "user_age": 30, "user_travelled_alone": "no", "user_embarked": "C"},
        {"user_pclass": 2, "user_sex": "female", "user_fare": 30.0, "user_age": 35, "user_travelled_alone": "yes", "user_embarked": "Q"},
        {"user_pclass": 2, "user_sex": "male", "user_fare": 20.0, "user_age": 40, "user_travelled_alone": "no", "user_embarked": "Q"},
        {"user_pclass": 3, "user_sex": "female", "user_fare": 10.0, "user_age": 45, "user_travelled_alone": "yes", "user_embarked": "S"},
        {"user_pclass": 3, "user_sex": "male", "user_fare": 5.0, "user_age": 50, "user_travelled_alone": "no", "user_embarked": "S"}
    ]

    for combo in combinations:
        response = client.post("/survivalchance", json={
            "user_pclass": combo["user_pclass"],
            "user_sex": combo["user_sex"],
            "user_fare": combo["user_fare"],
            "user_age": combo["user_age"],
            "user_travelled_alone": combo["user_travelled_alone"],
            "user_embarked": combo["user_embarked"],
            "user_prediction_model": "random_forest",
            "user_prediction_model_2": "knn",
            "user_prediction_model_3": "logistic_regression"
        })
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) == 3
        for res in response.json():
            assert "predicted probability of survival" in res

# Test the /survivalchance endpoint with extreme age and fare values combined
def test_survivalchance_extreme_values_combined():
    extreme_combinations = [
        {"user_age": 0.01, "user_fare": 0.01},
        {"user_age": 100.0, "user_fare": 1000.0}
    ]

    for combo in extreme_combinations:
        response = client.post("/survivalchance", json={
            "user_pclass": 1,
            "user_sex": "female",
            "user_fare": combo["user_fare"],
            "user_age": combo["user_age"],
            "user_travelled_alone": "no",
            "user_embarked": "C",
            "user_prediction_model": "random_forest",
            "user_prediction_model_2": "knn",
            "user_prediction_model_3": "logistic_regression"
        })
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) == 3
        for res in response.json():
            assert "predicted probability of survival" in res

# Test the /survivalchance endpoint with different valid model combinations
def test_survivalchance_valid_model_combinations():
    models = ["random_forest", "knn", "logistic_regression"]
    for model1 in models:
        for model2 in models:
            if model1 != model2:
                response = client.post("/survivalchance", json={
                    "user_pclass": 2,
                    "user_sex": "female",
                    "user_fare": 60.0,
                    "user_age": 40,
                    "user_travelled_alone": "yes",
                    "user_embarked": "Q",
                    "user_prediction_model": model1,
                    "user_prediction_model_2": model2,
                    "user_prediction_model_3": ""
                })
                assert response.status_code == 200
                assert isinstance(response.json(), list)
                assert len(response.json()) == 2
                for res in response.json():
                    assert "predicted probability of survival" in res
# Test the /survivalchance endpoint with minimal valid input
def test_survivalchance_minimal_input():
    response = client.post("/survivalchance", json={
        "user_pclass": 1,
        "user_sex": "female",
        "user_fare": 1.0,
        "user_age": 1,
        "user_travelled_alone": "yes",
        "user_embarked": "C",
        "user_prediction_model": "random_forest",
        "user_prediction_model_2": "",
        "user_prediction_model_3": ""
    })
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 1
    assert "predicted probability of survival" in response.json()[0]

# Test the /survivalchance endpoint with maximal valid input
def test_survivalchance_maximal_input():
    response = client.post("/survivalchance", json={
        "user_pclass": 3,
        "user_sex": "male",
        "user_fare": 1000.0,
        "user_age": 100,
        "user_travelled_alone": "no",
        "user_embarked": "S",
        "user_prediction_model": "random_forest",
        "user_prediction_model_2": "knn",
        "user_prediction_model_3": "logistic_regression"
    })
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 3
    for res in response.json():
        assert "predicted probability of survival" in res

# Test the /survivalchance endpoint with the same model used multiple times
def test_survivalchance_same_model_multiple_times():
    response = client.post("/survivalchance", json={
        "user_pclass": 2,
        "user_sex": "female",
        "user_fare": 75.0,
        "user_age": 35,
        "user_travelled_alone": "no",
        "user_embarked": "Q",
        "user_prediction_model": "random_forest",
        "user_prediction_model_2": "random_forest",
        "user_prediction_model_3": "random_forest"
    })
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 3
    assert "predicted probability of survival" in response.json()[0]

# Test the /survivalchance endpoint with null values in optional fields
def test_survivalchance_null_optional_fields():
    response = client.post("/survivalchance", json={
        "user_pclass": 2,
        "user_sex": "male",
        "user_fare": 50.0,
        "user_age": 40,
        "user_travelled_alone": "yes",
        "user_embarked": "Q",
        "user_prediction_model": "random_forest",
        "user_prediction_model_2": "",
        "user_prediction_model_3": ""
    })
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 1
    assert "predicted probability of survival" in response.json()[0]

# Test the /survivalchance endpoint with very low and very high ages together
def test_survivalchance_low_and_high_ages():
    response = client.post("/survivalchance", json={
        "user_pclass": 1,
        "user_sex": "female",
        "user_fare": 200.0,
        "user_age": 0.01,  # Newborn
        "user_travelled_alone": "no",
        "user_embarked": "C",
        "user_prediction_model": "random_forest",
        "user_prediction_model_2": "knn",
        "user_prediction_model_3": "logistic_regression"
    })
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 3
    for res in response.json():
        assert "predicted probability of survival" in res

    response = client.post("/survivalchance", json={
        "user_pclass": 1,
        "user_sex": "female",
        "user_fare": 200.0,
        "user_age": 100.0,  # Very old age
        "user_travelled_alone": "no",
        "user_embarked": "C",
        "user_prediction_model": "random_forest",
        "user_prediction_model_2": "knn",
        "user_prediction_model_3": "logistic_regression"
    })
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 3
    for res in response.json():
        assert "predicted probability of survival" in res
# Test the /survivalchance endpoint with a mixture of extreme high and low values
def test_survivalchance_mixed_extreme_values():
    response = client.post("/survivalchance", json={
        "user_pclass": 1,
        "user_sex": "male",
        "user_fare": 0.01,  # Very low fare
        "user_age": 99.9,  # Very high age
        "user_travelled_alone": "no",
        "user_embarked": "C",
        "user_prediction_model": "random_forest",
        "user_prediction_model_2": "knn",
        "user_prediction_model_3": "logistic_regression"
    })
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 3
    for res in response.json():
        assert "predicted probability of survival" in res

    response = client.post("/survivalchance", json={
        "user_pclass": 3,
        "user_sex": "female",
        "user_fare": 1000.0,  # Very high fare
        "user_age": 0.1,  # Very low age
        "user_travelled_alone": "yes",
        "user_embarked": "S",
        "user_prediction_model": "random_forest",
        "user_prediction_model_2": "knn",
        "user_prediction_model_3": "logistic_regression"
    })
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 3
    for res in response.json():
        assert "predicted probability of survival" in res
