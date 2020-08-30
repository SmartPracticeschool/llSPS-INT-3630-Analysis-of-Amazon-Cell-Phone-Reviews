import numpy as np
from flask import Flask, request,url_for, render_template
from keras.models import load_model
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tk=Tokenizer()
import tensorflow as tf
global graph 
graph = tf.get_default_graph()
filename = r"E:\Bunny\VEC-rsip\Project\Amazon.h5"
cla=load_model(filename)
with open(r'E:\Bunny\VEC-rsip\Project\cv_transform.pkl','rb') as file:
    cv=pickle.load(file)

cla.compile(optimizer='adam',loss='binary_crossentropy')
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        img_url = url_for('static',filename = 'css/style/0123.png')
        return render_template('index.html',url=img_url)
    if request.method == 'POST':
        data = request.form['message']
        print("Hey " +data)
        data=cv.transform([data])
        print("\n"+str(data.shape)+"\n")
        with graph.as_default():
            y_pred = cla.predict(data)
            print("pred is "+str(y_pred))       
        if(y_pred > 0.5):
            img_url = url_for('static',filename = 'css/style/static2.png')
            data = "Positive Review"
        else:
            img_url = url_for('static',filename = 'css/style/static1.jpg')
            print(img_url)
            data = "Negative Review"
            
        return render_template('index.html',prediction=data)        
if __name__ == '__main__':
    app.run(host = 'localhost', debug = True, threaded = False)
    
    
