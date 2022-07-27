from email import message
from hashlib import new
from pyexpat import model
import random
import json
import pickle
from statistics import mode
from unittest import result
import numpy as np

from datetime import date 

import webbrowser


import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from yaml import load


from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

import time


lemmatizer = WordNetLemmatizer()


import joblib

#for the first model named "model"
intents=json.loads(open('intents.json').read())

words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))



model=load_model('chatbotmodel.h5')


def clean_up_sentence(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words= [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words=clean_up_sentence(sentence)
    bag=[0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    bow=bag_of_words(sentence)
    res=model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD=0.25
    results=[[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x:x[1],reverse=True)
    return_list=[]
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    return return_list


def get_response(intents_list,intents_json):
    tag=intents_list[0]['intent']
    list_of_intents=intents_json['intents']
    for i in list_of_intents:
        if i['tag']==tag:
            result=random.choice(i['responses'])
            break
    return result
#for the second model named "modelf"
intentsf=json.loads(open('intentsf.json').read())

wordsf=pickle.load(open('wordsf.pkl','rb'))
classesf=pickle.load(open('classesf.pkl','rb'))



modelf=load_model('chatbotmodelf.h5')


def clean_up_sentencef(sentence):
    sentence_wordsf=nltk.word_tokenize(sentence)
    sentence_wordsf= [lemmatizer.lemmatize(word) for word in sentence_wordsf]
    return sentence_wordsf

def bag_of_wordsf(sentence):
    sentence_wordsf=clean_up_sentencef(sentence)
    bag=[0]*len(wordsf)
    for w in sentence_wordsf:
        for i,word in enumerate(wordsf):
            if word == w:
                bag[i]=1
    return np.array(bag)

def predict_classf(sentence):
    bow=bag_of_wordsf(sentence)
    res=modelf.predict(np.array([bow]))[0]
    ERROR_THRESHOLD=0.25
    results=[[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x:x[1],reverse=True)
    return_list=[]
    for r in results:
        return_list.append({'intent':classesf[r[0]],'probability':str(r[1])})
    return return_list


def get_responsef(intentsf_list,intentsf_json):
    tag=intentsf_list[0]['intent']
    list_of_intentsf=intentsf_json['intentsf']
    for i in list_of_intentsf:
        if i['tag']==tag:
            result=random.choice(i['responses'])
            break
    return result

#for the third model named "modela"
intentsa=json.loads(open('intentsa.json').read())

wordsa=pickle.load(open('wordsa.pkl','rb'))
classesa=pickle.load(open('classesa.pkl','rb'))



modela=load_model('chatbotmodela.h5')


def clean_up_sentencea(sentence):
    sentence_wordsa=nltk.word_tokenize(sentence)
    sentence_wordsa= [lemmatizer.lemmatize(word) for word in sentence_wordsa]
    return sentence_wordsa

def bag_of_wordsa(sentence):
    sentence_wordsa=clean_up_sentencea(sentence)
    bag=[0]*len(wordsa)
    for w in sentence_wordsa:
        for i,word in enumerate(wordsa):
            if word == w:
                bag[i]=1
    return np.array(bag)

def predict_classa(sentence):
    bow=bag_of_wordsa(sentence)
    res=modela.predict(np.array([bow]))[0]
    ERROR_THRESHOLD=0.25
    results=[[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x:x[1],reverse=True)
    return_list=[]
    for r in results:
        return_list.append({'intent':classesa[r[0]],'probability':str(r[1])})
    return return_list


def get_responsea(intentsa_list,intentsa_json):
    tag=intentsa_list[0]['intent']
    list_of_intentsa=intentsa_json['intentsa']
    for i in list_of_intentsa:
        if i['tag']==tag:
            result=random.choice(i['responses'])
            break
    return result




#get the user name
text = input("")

def tell_me_my_name(text):
    nltk_results = ne_chunk(pos_tag(word_tokenize(text)))
    for nltk_result in nltk_results:
        if type(nltk_result) == Tree:
            name = ''
            for nltk_result_leaf in nltk_result.leaves():
                name += nltk_result_leaf[0] + ' '
            return(name)

#create a file .txt , it will be used for sentiments_analysis

file = open('sentiment_analysis.txt','a+') 

""" verifier si une chaine de caractere appartient a un fichier  """

def exist(text):
    file=open('sentiment_analysis.txt','r')
    lines=file.readlines()
    lines=[x.replace('\n','') for x in lines]
    lines=[i for i in lines if i!='']
    for i in lines:
        if(text in i):
            return(True)
    return(False)


#fucntion that find url in text data
from urlextract import URLExtract
def yes_url(sentence):
    extractor = URLExtract()
    urls = extractor.find_urls(sentence)
    return(urls!=[])

    return()
def extract_url(sentence):
    extractor = URLExtract()
    urls = extractor.find_urls(sentence)
    return((sentence.replace(urls[0],""),urls[0]))
    

#predict function for language detection

pipe=joblib.load("pipe.joblib")

def predict_language(text):
    lang = pipe.predict([text])
    return(lang[0])


def language_score(msg):
    probability=pipe.predict_proba([msg])
    return(np.max(probability[0]))


#main


def final_response(message):
    
    if((predict_language(message)=="English")):
        #we predict first the intent(the class) from the message
        ints=predict_class(message)
        #we create a file that contains responses from user (it will be used for sentiment analysis)
        file = open("sentiment_analysis.txt", "a")
        if(not(exist(str(date.today())))):
            file.write("\n"+str(date.today()))
        file.write("\n"+message)
        file.close()
        #if the user introduce himself with his name, we gonna call him with it
        if(ints[0]['intent']=='MY_NAME_IS'):
            res=get_response(ints,intents)
            name=tell_me_my_name(message)
            if(name!=None):
                return(res+ " " +name)
            else:
                return("Welcome")
        #if we predict that the user answer for the date today
        elif(ints[0]['intent']=='Date'):
            return("It's "+ str(date.today()))
        else:
            #if the response in the intents file contains an url so we will open it.
            if(yes_url(get_response(ints,intents))):
                url=extract_url(get_response(ints,intents))
                

                webbrowser.open(url[1],new=0)
                return ("we hope that this was helpful, if you have any problem don't hestitate to contact us! ")
            else:
                
                return get_response(ints,intents)
    elif(predict_language(message)=="French"):
        intent=predict_classf(message)
        return(get_responsef(intent,intentsf))
    elif(predict_language(message)=="Arabic"):
        intent=predict_classa(message)
        return(get_responsea(intent,intentsa))



    





 