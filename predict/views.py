from django.http import response
from django.shortcuts import render
import pickle
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from django.http import HttpResponse
# Create your views here.
import requests
from django.views.decorators.csrf import csrf_exempt

d={}

def home(request):
    return render(request,"home.html")
d={}
@csrf_exempt
def chatbot(request):
    if request.method == 'GET':            
        inp= request.GET.get('inp')
    with open("predict/static/New_intents.json") as file:
        data = json.load(file)
    chat_model = load_model('predict/static/models/LSTM_Chatbot.h5')
    # load tokenizer object
    with open('predict/static/models/token.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('predict/static/models/encoded.pickle', 'rb') as enc:
        onehot_encoded = pickle.load(enc)
    max_len = 10
    result = chat_model.predict(pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
    print(result)
    print(np.argmax(result))

    emotion=["depressed","Greeting","Goodbye_Thanks","relationships","followup_relationships","studies","followup_studies","family","followup_family"]    
    tag=emotion[np.argmax(result)]
    for i in data['intents']:
        if i['tag'] == tag:
            a=np.random.choice(i['responses'])
            d[inp]=a
            return render(request,"home.html",{"d":d})
