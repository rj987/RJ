from flask import Flask,render_template,request
from flask_cors import cross_origin
import sklearn
import pickle
import numpy as np

from sys import stderr


app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
col=['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Area Population']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST',"GET"])
def predict():
    int_feature=[int[x] for x in request.form.values()]
    final=[np.array(int_feature,dtype=float)]
    prediction=model.predict(final)
    output=round(prediction[0,2])
    
    return render_template('index.html', pred='The price of your dream house is {} USD Only.'.format(output))


if __name__=='__main__':
    app.run(debug=True)
    