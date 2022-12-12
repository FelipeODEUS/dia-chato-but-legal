import nltk
import json
import pickle
import numpy as np
import random
import tensorflow
from data_preprocessing import get_stem_words

ignore_words = ['?', '!',',','.', "'s", "'m"]
classif = tensorflow.keras.models.load_model("./chatbot_model.h5")
intents = json.loads(open("./intents.json").read())
words = pickle.load(open("./words.pkl", "rb"))
classes = pickle.load(open("./classes.pkl", "rb"))

def funssaom(user_input):
    config1 = nltk.word_tokenize(user_input)
    config2 = get_stem_words(config1, ignore_words)
    config2 = sorted(list(set(config2)))
    bag = []
    bag_of_words = []
    for word in words:            
        if word in config2:              
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    bag.append(bag_of_words)
    return np.array(bag)

def funssaom2(user_input):
    saida = funssaom(user_input)
    prevision = classif.predict(saida)
    end_prevision = np.argmax(prevision[0])
    return end_prevision

def funssaom3(user_input):
    predicted_class_label =  funssaom2(user_input)
    predicted_class = classes[predicted_class_label]

    for intent in intents['intents']:
        if intent['tag']==predicted_class:
            bot_response = random.choice(intent['responses'])
            return bot_response

print("Oi, eu sou a Estela, como posso ajudar?")

while True:
    user_input = input("Digite sua mensagem aqui:")
    print("Entrada do Usuário: ", user_input)

    response = funssaom3(user_input)
    print("Resposta do Robô: ", response)