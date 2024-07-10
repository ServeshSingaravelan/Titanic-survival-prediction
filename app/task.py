from queue import Full
from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9090"],  # need to adjust the origin to React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# from userdataretrieval import user_pclass,user_age,user_fare,user_embarked,user_prediction_model,user_sex,user_travelled_alone,user_prediction_model_2,user_prediction_model_3
from app.predictornew import predict_survival


#storage to hold user data
user_data_store = {}
next_user_id = 1  # Initialize the next user ID to 1

# class User(BaseModel):
#     pclass: str
#     sex: str
#     age: int
#     fare: int
#     travelled_alone: Union[bool, None] = None
#     embarked: str
#     prediction_model: str
# survival_rate=predict_survival(user_pclass, user_sex, user_fare, user_age, user_travelled_alone, user_embarked, user_prediction_model,user_prediction_model_2,user_prediction_model_3)

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}
# # @app.get("/survivalchance")
# # def survival_chance():
# #     return(survival_rate) #probability of the model
# @app.post("/survivalchance")
# def survivalchance(user_pclass, user_sex, user_fare, user_age, user_travelled_alone, user_embarked, user_prediction_model,user_prediction_model_2,user_prediction_model_3):
#     #  {
#     #     "user_class": user_pclass,
#     #     "embarked": user_embarked,
#     #     "age": user_age,
#     #     "fare": user_fare,
#     #     "sex": user_sex,
#     #     "travelled_alone": user_travelled_alone,
#     #     "prediction_model": user_prediction_model,
#     #     "prediction_model_2": user_prediction_model_2,
#     #     "prediction_model_3": user_prediction_model_3
#     # }
#      return predict_survival(user_pclass, user_sex, user_fare, user_age, user_travelled_alone, user_embarked, user_prediction_model,user_prediction_model_2,user_prediction_model_3)

    
# # @app.get("/users/{user_id}")
# # def read_item(user_id: int):
# #     if user_id in user_data_store:
# #         return user_data_store[user_id]
# #     else:
# #         raise HTTPException(status_code=404, detail="User not found")


# # @app.post("/users")
# # def add_user(user: User):
# #      {
# #         "user_class": user.pclass,
# #         "age": user.age
# #         "fare": user.fare,
# #         "sex": user.sex,
# #         "travelled_alone": user.travelled_alone,
# #         "embarked": user.embarked,
# #         "prediction_model": user.prediction_model
# #     }
# #     return user_data_store[next_user_id]
class SurvivalRequest(BaseModel):
    user_pclass: int
    user_sex: str
    user_fare: float
    user_age: float
    user_travelled_alone: str
    user_embarked: str
    user_prediction_model: str
    user_prediction_model_2: str
    user_prediction_model_3: str


@app.post("/survivalchance")
def survivalchance(data: SurvivalRequest):

    return predict_survival(
        data.user_pclass, data.user_sex, data.user_fare, 
        data.user_age, data.user_travelled_alone, data.user_embarked,
        data.user_prediction_model, data.user_prediction_model_2,
        data.user_prediction_model_3
    )
