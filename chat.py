import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Stephane"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    def recherche():
        from googlesearch import search

        
        for i in search(msg,num_results=10, lang="fr"):
            #print("\n")
            #print("\n")
            #print("une liste de lien s'afficherons pour fous indiquer ou regarder  ")
            #print("\n")
            #print("\n")
            #print("  ==>>   ",i)
            #print("\n")
            #print("\n")
            return "Desolé mais j'ai pas compris ce que vous dites Reformulez svp            ou bien retirer l'accent            et veillez notez juste les mots clé svp exemple >>Hebergement<< sans accent des fois  :\n"
            #return "Desolé mais j'ai pas compris ce que vous dites Reformulez svp            ou bien retirer l'accent            merci :\n                      ce si est la liste des site ou vous pourez trouver ce que vous cherchez                         \n                    ",i,"     \n"
    
    
    return recherche() 


if __name__ == "__main__":
    print("commencon le  chat! (tape 'quit' pour sortir)")
    while True:
        # 
        sentence = input("client: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print("steph :===>",resp)

