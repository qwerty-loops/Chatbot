#Main purpose of this python file is to train the chatbot

#STEP 1
#Import libraries for general purpose
import numpy as np
import pandas as pd
import random

#Keras is a deep learning API developed by Google
#It is used for implementing neural networks
#It supports multiple backend neural network computation

from keras.models import Sequential #The Sequential API arranges Keras layers in a sequential order.
from keras.optimizers import SGD #It is an optimization algorithm commonly used in deep learning and is designed to update the weights of a neural network during training.
from keras.layers import Dense, Activation, Dropout 

# The Dense layer is a fully connected layer, ie every node in the layer is connected to every node in the previous and subsequent layers
#The Activation layer applies an activation function to its input
#The Dropout layer randomly sets a fraction of input units to 0

import nltk #This is a Natural Language Processing toolkit
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json #Provides encoding and decoding for Json data
import pickle #Provides us access to read, write data into text files

#Opening a json file and reading it
intents_file = open(r'D:\Allen Archive\Allen Archives\NEU_academics\Semester1\Python_notebooks\Semester 1\Self_projects\Chatbot\intents.json').read()
intents = json.loads(intents_file)

#######################     END OF STEP 1 ####################
#STEP 2
#Preprocessing the data 

#We shall be looking into 2 preprocessing techniques
#1.Tokenizing : Here we break sentences into words. We shall be tokenizing patterns and adding the words in a list.
#2.Lemmatizing : First convert words to lemma form.We can reduce the number of words in our vocab this way. By this way we can also get rid of duplicates.

#1. Firstly looking into Tokenizing
words=[]
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        word=nltk.word_tokenize(pattern)
        words.extend(word)        
        #add documents in the corpus
        documents.append((word, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
print(documents)

# #Alternately

# for w in words:
#     if w not in ignore_letters:
#         words=[lemmatizer.lemmatize(w.lower())]
#         #Sort words
#         #Set function removes duplicates, list function stores the set into a list and sorted is used to sort the remaining words of the list
#         words=sorted(list(set(words)))
#         # sort classes
#         classes = sorted(list(set(classes)))
#         # documents = combination between patterns and intents
#         print (len(documents), "documents")
#         # classes = intents
#         print (len(classes), "classes", classes)
#         # words = all words, vocabulary
#         print (len(words), "unique lemmatized words", words)
# pickle.dump(words,open('words.pkl','wb'))
# pickle.dump(classes,open('classes.pkl','wb'))

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#######################     END OF STEP 2 ####################

#Step 3 : Create Training and Testing Data

#Lemmatize each words
# and create a list of zeroes of the same length as the total number of words
#We will set value 1 to only those indexes that contain the word in the patterns. 
#In the same way, we will create the output by setting 1 to the class input the pattern belongs to

#Creating the training data
training=[]
#Create an empty array for output
output=[0]*len(classes)

#For the training set, bag of words for each sentence
for doc in documents:
    #Initializing the bag of words
    bag=[]
    # list of tokenized words for the pattern
    word_patterns=doc[0]
    #Lemmatizing each words : To find the base word
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # create the bag of words array with 1, if word is found in current pattern
    for word in words:
        if word in word_patterns:
            bag.append(1)
        else:
            bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# shuffle the features and make numpy array
random.shuffle(training)
Data_type = object
training = np.array(training, dtype=Data_type)
#training = np.array(training)
# create training and testing lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data is created")

#Step 4 : Training the model

#First layer has 128 neurons
#Second layer has 64 neurons
#Third layer has as many neurons as the number of classes
#The dropout layers are introduced to reduce overfitting of the model
#We have used the SGD optimizer and fit the data to start the training of the model
#Post triaining of 200 epochs we save the trained model using the Keras model.save  function.
# deep neural networds model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
# Compiling model. SGD with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#Training and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("model is created")