from flask import Flask, render_template, request
#import numpy as np
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World'


@app.route('/index', methods=["POST", "GET"])
def index():
    return render_template('index.html')


@app.route('/test', methods=["POST"])
def index1():
    result = request.get_json()
    print(result["msg"], flush=True)
    return result["msg"]


'''
def spamCheckFunction():
    V = ["secret", "offer", "low", "price", "valued", "customer", "today",
         "dollar", "million", "sports", "is", "for", "play", "healthy", "pizza"]
    spam_msgs = ["million dollar offer",
                 "secret offer today", "secret is secret"]
    ham_msgs = ["low price for valued customer offer",
                "play secret sports today", "sports is healthy", "low price pizza"]
    v_idx_mapping = {}
    for i, word in enumerate(V):
        v_idx_mapping[word] = i

    table = np.zeros([2, len(V)])

    spam_list = [set(spam_msg.split(' ')) for spam_msg in spam_msgs]
    ham_list = [set(ham_msg.split(' ')) for ham_msg in ham_msgs]
    for i, word in enumerate(V):
       for spam_msg in spam_list:
          table[0][i] += word in spam_msg     
       for ham_msg in ham_list:
          table[1][i] += word in ham_msg
          
         
    num_spam = len(spam_msgs)
    num_ham = len(ham_msgs)
    num_total = num_ham + num_spam
    
    return ""
'''

if __name__ == '__main__':
    app.run('0.0.0.0')
