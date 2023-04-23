#Import necessary libraries
from flask import Flask, render_template, request

import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

filepath = "D:/PROJECT/NEOPLASM/model.h5"
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")

def pred_barin_neoplasm(brain_img):
    img = image.load_img(brain_img, target_size=(64, 64)) # load image 
    print("@@ Got Image for prediction")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0]
  
    print(pred)
    if pred==0:
        return "No Neoplasm Detected", 'No Tumor.html'
       
    elif pred==1:
        return "Neoplasm Detected", 'Tumor.html'
        

    

# Create flask instance
app = Flask(__name__)

# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
    
 
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('D:/PROJECT/NEOPLASM/static/upload/', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = pred_barin_neoplasm(brain_img=file_path)
              
        return render_template(output_page, pred_output = pred, user_image = file_path)
    
# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False,port=8000) 
    
    
