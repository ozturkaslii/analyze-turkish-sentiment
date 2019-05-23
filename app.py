from flask import Flask, render_template, request
import tensorflow as tf
import pickle
from keras.models import load_model
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.backend import clear_session
import pandas as pd

clear_session()

app = Flask(__name__)

# load the Model from file
nlp_model = load_model('lstm_nlp1.h5')

global graph
graph = tf.get_default_graph()

# load tokenizer
with open('turkish_tokenizer.pickle', 'rb') as handle:
    turkish_tokenizer = pickle.load(handle)

def predict(texts):
	tokens = turkish_tokenizer.texts_to_sequences(texts)
	tokens_pad = pad_sequences(tokens, maxlen=59)
	with graph.as_default():
		prediction = nlp_model.predict(tokens_pad)[0][0]
	return prediction


@app.route('/', methods=['GET', 'POST'])
def home():
	in_text = request.values.get('text_input')
	arr = [in_text]
	# if input is provided process else show default page
	if request.method == 'POST':
		result = predict(arr)
		return render_template('home.html', result=result, text=in_text)
	else:
		return render_template('home.html')


if __name__ == '__main__':
	app.run(debug=False)
