from flask import Flask, request, render_template
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import nltk

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)

cv = pickle.load(open('vocabulary.pkl', 'rb'))  
model = joblib.load('Sentiment_Model_trained')

stop_words_new = stopwords.words('english')
stemmer = PorterStemmer()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_text = "" 

    if request.method == "POST":
        text = request.form['text']
        input_text = text  

        #clean the input
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower().split()
        text = [stemmer.stem(word) for word in text if word not in stop_words_new]
        text = ' '.join(text)
        print(text)

        #vectorize the input text and make the prediction
        text_vectorized = cv.transform([text])
        prediction = model.predict(text_vectorized)

        result = 'Positive' if prediction == 1 else 'Negative'

    return render_template("index.html", result=result, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)
