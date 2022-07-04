from flask import Flask, request, render_template
from flask_cors import cross_origin
from werkzeug.utils import secure_filename
import sklearn
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input 
import cv2
import os
import numpy as np

app = Flask(__name__,template_folder='template',static_url_path='/static')
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")

@app.route('/mlpredict')
def mlpredict():
    return render_template("b.html",name='display')

@app.route('/dlpredict')
def dlpredict():
    return render_template("c.html",name='display')

@app.route('/predicml',methods =["GET", "POST"])
def predicml():
    if request.method == "POST":
        mmse = request.form.get("mmse")
        cdr = request.form.get("cdr")
        memory = request.form.get("memory")

        mmse=float(mmse)
        cdr = float(cdr)
        memory = float(memory)
        

        with open('alzheimers.pkl', 'rb') as file:
            data = pickle.load(file)

        svmclassifier = data["model"]  

        prediction = svmclassifier.predict(np.array([[memory,cdr,mmse]]))  


        return render_template("b.html",prediction=prediction[0])

@app.route('/predicdl',methods =["GET", "POST"])
def predicdl():
    classes = {0:'MildDemented',1: 'ModerateDemented',2:'NonDemented',3:'VeryMildDemented'}
    if request.method == 'POST':
        if 'file' in request.files:
            f = request.files['file']
            print(f)
            f.save('./static/imgs/'+secure_filename(f.filename))


            dlmodel = load_model('model-ep45-val_loss1.605.h5')
            X = cv2.imread('./static/imgs/'+secure_filename(f.filename))
            X=cv2.resize(X,(176,176))
            X = preprocess_input(X)
            X=X.reshape(1,176,176,3)
            max_index = np.argmax(dlmodel.predict(X)[0])
            
            return render_template("c.html",prediction=classes[max_index])
            

        else:
            return render_template("index.html")   



if __name__ == '__main__':
    app.debug = True
    app.run()