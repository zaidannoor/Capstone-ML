import random
import json
import tensorflow as tf
import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

from flask import Flask, request

app = Flask(__name__)

class NeuralNet(tf.keras.Model):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.l2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.l3 = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        return out

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

with open('data_intents.json', 'r') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?', '!']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stem(w) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

bot_name = "SMILE"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, words)
    X = np.array([X])
    
    output = model(X)
    predicted_class = np.argmax(output)
    tag = classes[predicted_class]

    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    return "I do not understand..."

# Create an instance of the model
model = NeuralNet(100, 100, len(intents['intents']))
tf.keras.models.load_model('model2.h5')

@app.route("/get_response", methods=["POST"])
def get_bot_response():
    data = request.json
    question = data.get("question")
    response = get_response(question)
    return {"response": response}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6060)