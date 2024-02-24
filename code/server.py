# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     server
   Author :       huangkai
   date：          2024/2/24
-------------------------------------------------
"""
import uvicorn
from fastapi import FastAPI
import joblib
from os.path import dirname, join, realpath
from typing import List

app = FastAPI(
    title="Iris Prediction Model API",
    description="A simple API that use LogisticRegression model to predict the Iris species",
    version="0.1",
)

# load  model

with open(
        join(dirname(realpath(__file__)), "models/IrisClassifier.pkl"), "rb"
) as f:
    model = joblib.load(f)


def data_clean(str):
    arr = str.split(',')
    arr = list(map(float, arr))
    return arr


# Create Prediction Endpoint
@app.get("/predict-result")
def predict_iris(request):
    # perform prediction
    request = data_clean(request)
    prediction = model.predict([request])
    output = int(prediction[0])
    probas = model.predict_proba([request])
    output_probability = "{:.2f}".format(float(probas[:, output]))

    # output dictionary
    species = {0: "Setosa", 1: "Versicolour", 2: "Virginica"}

    # show results
    result = {"prediction": species[output], "Probability": output_probability}
    return result


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8001)