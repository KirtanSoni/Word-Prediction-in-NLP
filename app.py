from cgitb import text
import os
from flask import Flask,render_template, request,jsonify
from predict import predict_next_words
import pickle
from tensorflow.keras.models import load_model


with open('token/tokenizer1.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model('nextword3.h5')


photos = os.path.join('static', 'image')
svg = os.path.join('static', 'svgs')

app = Flask(__name__)

app.static_folder = 'static'
app.config['img_folder'] = photos 
app.config['svg_folder'] = svg

txt = ""
req_dict = None


@app.route('/about_us')
def about_us():
    devimage = os.path.join(app.config['img_folder'], 'ak.jpg')
    fblogo = os.path.join(app.config['svg_folder'], 'facebook-brands.svg')
    linkedinlogo = os.path.join(app.config['svg_folder'], 'linkedin-brands.svg')
    githublogo = os.path.join(app.config['svg_folder'], 'github-brands.svg')
    return render_template("about_us.html", devimage=devimage, facebook=fblogo, linkedin=linkedinlogo, github=githublogo)


@app.route("/")
def home():
    return render_template("Home.html")
    

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    global req_dict,req_json
    if request.method == "POST":
        req_dict=request.get_json()
        predict=[]
        predict = req_dict['str'].split("_")
        predict = predict_next_words(model, tokenizer, predict)
        print(predict)
        return predict
    else:
        return "need more words"


if __name__ == "__main__":
    app.run(debug=True)
