from cgitb import text
from flask import Flask,send_file
from flask import request,jsonify
from flask import Flask, render_template, request
from predict import predict_next_words
import string
import pickle
from tensorflow.keras.models import load_model


with open('token/tokenizer1.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model('nextword3.h5')


app = Flask(__name__)
txt = ""
req_dict=None 
req_json=None

@app.route("/", methods=['GET', 'POST'])
def home():
    global req_dict,req_json
    '''if(request.method=='POST'):
        req_dict=dict(request.form)
        req_json=jsonify(req_dict)
        print(req_dict)
    return render_template("Home.html", prediction="Need more words")'''
    if request.method == "POST":
        global txt
        form  =  request.form
        predict = form['input_temp']
        txt = txt + predict + " "
        if predict == "":
            return render_template("Home.html", prediction="Need more words")
        predict = predict.split()[-3:]
        predict = predict_next_words(model, tokenizer, predict)
        print(predict)
        return render_template("Home.html", prediction=predict)
    else:
        return render_template("Home.html", prediction="Need more words")


if __name__ == "__main__":
    app.run(debug=True)
