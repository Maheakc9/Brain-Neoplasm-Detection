from flask import Flask,render_template
from flask import Flask, render_template, request

import numpy as np
import os

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

filepath = "D:/PROJECT/NEOPLASM/model.h5"
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")


app = Flask(__name__)

@app.route('/', methods=['GET'])

def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=8000,debug = False)