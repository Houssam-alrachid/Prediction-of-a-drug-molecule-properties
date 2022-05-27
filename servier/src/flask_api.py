
from flask import Flask, request
import numpy as np
import pandas as pd
import flasgger
from flasgger import Swagger # Creat a FrontEnd
from servier.src.main import Predict


app=Flask(__name__) # from where you want to start this API
Swagger(app)


@app.route('/') # a decorator : my route API : welcome page
def welcome():
    return "Welcome All"

@app.route('/predict_from_file/<path:path_X_test>')
def predict_from_file(path_X_test):
    """Let's predict basic molecule properties.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    
    # Example: 
    # path_X_test=servier/data/dataset_single_test.csv
    # Link: 
    #     http://10.188.219.126:8000/predict_from_file/servier/data/dataset_single_test.csv
    # OR :
    #     http://10.188.219.126:8000/predict_from_file/servier/data/dataset_single.csv 

    y_pred = Predict(path_X_test, model_type = 'FF')
    return {'Hello the answer is y_pred': y_pred.tolist()}

# @app.route('/predict_file',methods=["POST"])
# def predict_note_file():
#     """Let's predict basic molecule properties.
#     ---
#     parameters:
#       - name: file
#         in: formData
#         type: file
#         required: true
      
#     responses:
#         200:
#             description: The output values
        
#     """
#     X_test=pd.read_csv(request.files.get("file"))
#     y_pred =Predict(X_test, model_type = 'FF')
    
#     return str(list(y_pred))


if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)
    