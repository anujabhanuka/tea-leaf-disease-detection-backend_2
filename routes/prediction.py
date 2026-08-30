import os

from flask import request
from flask import render_template

from app import app

from utils.predictor import predict
from utils.deaseas_info import DISEASE_INFO

@app.route("/predict", methods=["POST"])

def prediction():

    file = request.files["image"]

    path = os.path.join("static/uploads", file.filename)

    file.save(path)

    disease, confidence = predict(path)

    description = DISEASE_INFO[disease]

    return render_template(
        "result.html",
        disease=disease,
        confidence=round(confidence*100,2),
        description=description,
        image=path
    )