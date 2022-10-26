from flask import Flask, render_template, request
from flask_jsonpify import jsonpify
import pandas as pd
from model.nbModel import nbModel 
#import numpy as np
app = Flask(__name__)

nmodel=nbModel()

@app.route('/')
def hello_world():
    df = pd.read_csv('static/data.csv', dtype=str)
    html = df.to_html()
    return html
    

@app.route('/index', methods=["POST", "GET"])
def index():
    return render_template('index.html')


@app.route('/result', methods=["POST"])
def index1():
    result = request.get_json()
    res = nmodel.question(result["msg"])
    rmsg = ""
    if res == 1:
        rmsg = "1#스미싱 입니다. 주의하세요"
    elif res == 0:
        rmsg = "2#정상 문자 입니다."
    
    return rmsg


@app.route('/train', methods=["GET"])
def spamCheckFunction():
    return nmodel.training()
   
   

if __name__ == '__main__':
    app.run('0.0.0.0')
