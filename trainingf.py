import random
import json
import pickle
from statistics import mode
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation , Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()



intentsf=json.loads(open('intentsf.json').read())

wordsf=[]
classesf=[]
documentsf= []
ignore_letters=['?','!','.',',']#we can ignore this


for intent in intentsf['intentsf']:
    for pattern in intent['patterns']:
        word_list=nltk.word_tokenize(pattern)
        wordsf.extend(word_list)
        #wordsf contain all 
        documentsf.append((word_list, intent['tag']))
        if(intent['tag'] not in classesf):
            classesf.append(intent['tag'])



wordsf=[lemmatizer.lemmatize(word) for word in wordsf if word not in ignore_letters]
wordsf=sorted(set(wordsf))
classesf=sorted(set(classesf))

pickle.dump(wordsf,open('wordsf.pkl','wb'))
pickle.dump(classesf,open('classesf.pkl','wb'))




training=[]
output_empty=[0]*len(classesf)

for document in documentsf:
    bag=[]
    word_patterns=document[0]
    word_patterns=[lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in wordsf:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row=list(output_empty)
    output_row[classesf.index(document[1])]=1
    training.append([bag,output_row])

random.shuffle(training)
training=np.array(training)

training_x=list(training[:,0])
#167
training_y=list(training[:,1])
#24

modelf=Sequential()
modelf.add(Dense(128,input_shape=(len(training_x[0]),),activation="relu"))
modelf.add(Dropout(0.5))
modelf.add(Dense(64,activation='relu'))
modelf.add(Dropout(0.5))
modelf.add(Dense(len(training_y[0]),activation="softmax"))

sgd=SGD(learning_rate=0.01,decay=1e-6,momentum=0.9,nesterov=True)
modelf.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])


hist=modelf.fit(np.array(training_x),np.array(training_y),epochs=200,batch_size=5,verbose=1)
modelf.save('chatbotmodelf.h5',hist)










