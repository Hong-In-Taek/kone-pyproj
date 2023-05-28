from flask import Flask, render_template, request
from flask_jsonpify import jsonpify
import pandas as pd
from model.kobert import kobertMbti


#import numpy as np
app = Flask(__name__)

kobertmbti = kobertMbti()


@app.route('/', methods=["POST", "GET"])
def index():
    return render_template('index.html')


@app.route('/mbti/result', methods=["POST"])
def mbtiresultFunction():
    result = request.get_json()
    res = kobertmbti.mbtiFuction(result["msg"])
    print(res, flush=True)

    return res


@app.route('/mbti/train', methods=["GET"])
def mbtiTrainingFunction():
    return mbtinpmodel.train()


@app.route('/popup', methods=['GET'])
def popup():
    mbtivalue = request.args.get('mbti')  # 입력된 텍스트 값 받아오기
    imageUrl = "static/"+mbtivalue.upper()+".png"
    return render_template('popup.html', image_url=imageUrl)


if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
