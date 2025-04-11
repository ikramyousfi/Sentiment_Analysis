from flask import Flask, request, render_template
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import nltk

# Ensure stopwords are downloaded only once
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load models and vocabulary
cv = pickle.load(open('vocabulary.pkl', 'rb'))  
model = joblib.load('Sentiment_Model_trained')

# Define other necessary variables
stop_words_new = stopwords.words('english')
stemmer = PorterStemmer()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_text = ""  # Default empty string for text input

    if request.method == "POST":
        # Get the input text from the form
        text = request.form['text']
        input_text = text  # Save the input text for later display

        # Preprocess the input text
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower().split()
        text = [stemmer.stem(word) for word in text if word not in stop_words_new]
        text = ' '.join(text)

        # Vectorize the text and make the prediction
        text_vectorized = cv.transform([text])
        prediction = model.predict(text_vectorized)

        # Display the result
        result = 'Positive' if prediction == 1 else 'Negative'

    return render_template("index.html", result=result, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)
