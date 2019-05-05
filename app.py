from flask import Flask, render_template, request
import tensorflow as tf
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

dataset = pd.read_csv('ecommercereviews.csv')
data = dataset['Review'].values.tolist()


def predict(texts):
	tokenizer = Tokenizer(num_words=10000)
	tokenizer.fit_on_texts(data)
	tokens = tokenizer.texts_to_sequences(texts)
	tokens_pad = pad_sequences(tokens, maxlen=59)
	with graph.as_default():
		prediction = nlp_model.predict(tokens_pad)[0][0]
	return prediction


@app.route('/')
def home():
	returning_html = ''
	in_text = request.args.get('text_input')
	arr = [in_text]
	# if input is provided process else show default page
	if in_text is not None:
		returning_html += in_text
		result = predict(arr)
		if result > 0.5:
			returning_html += '<h4> Positive Sentiment :)</h4><br>'
		else:
			returning_html += '<h4> Negative Sentiment :(</h4><br>'
		return returning_html
	else:
		return render_template('home.html')


if __name__ == '__main__':
	app.run(debug=False)
