from flask import Flask, render_template, request
from flask_jsonpify import jsonpify
import pandas as pd
from model.nbModel import nbModel
from model.mbtiNPmodel import mbtiNPmodel


#import numpy as np
app = Flask(__name__)

nmodel = nbModel()
mbtinpmodel = mbtiNPmodel()


@app.route('/data')
def hello_world():
    df = pd.read_csv('static/mbtiData.csv', dtype=str)
    html = df.to_html()
    return html


@app.route('/', methods=["POST", "GET"])
def index():
    return render_template('index.html')


'''
@app.route('/result', methods=["POST"])
def index1():
    result = request.get_json()
    res = nmodel.question(result["msg"])
    rmsg = ""
    if res == 1:
        if kbt.Analyze_data_sentencetest(result["msg"], finetuned=True) < 5:
            if rnn.analyze_test_data(result["msg"]) > 80:
                rmsg = "1#스미싱 입니다. 주의하세요"
            else:
                rmsg = "2#정상 문자 입니다."
        else:
            rmsg = "2#정상 문자 입니다."
    elif res == 0:
        rmsg = "2#정상 문자 입니다."

    return rmsg


@app.route('/train', methods=["GET"])
def spamCheckFunction():
    return nmodel.training()
'''


@app.route('/mbti/result', methods=["POST"])
def mbtiresultFunction():
    result = request.get_json()
    res = mbtinpmodel.tellmemyMBTI(result["msg"], "hit")
    print(res, flush=True)

    return res


@app.route('/mbti/train', methods=["GET"])
def mbtiTrainingFunction():
    return mbtinpmodel.train()


if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
