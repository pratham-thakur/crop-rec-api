from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Crop_rec(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# loading the saved model
model = pickle.load(open('LogisticRegression.pkl', 'rb'))


@app.post('/predict')
def pred(input_parameters: Crop_rec):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    n = input_parameters.N
    p = input_parameters.P
    k = input_parameters.K
    temp = input_parameters.temperature
    hum = input_parameters.humidity
    ph = input_parameters.ph
    rain = input_parameters.rainfall

    input_list = [n, p, k, temp, hum, ph, rain]
    
    prediction = model.predict([input_list])

    return {'prediction': prediction[0]}


