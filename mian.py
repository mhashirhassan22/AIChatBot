from tkinter import *
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tensorflow as tf
import random
import keras
import pickle
import xlrd
import json
import pandas

root = Tk()
root.title("ChatBot")

frame= Frame(root,bg="black",width=580)
canvas = Canvas(frame,bg="black",width=580)
scrollbar = Scrollbar(frame, orient="vertical", command=canvas.yview)
chatwindow = Frame(canvas,bg="black")

chatwindow.bind("<Configure>",lambda e: canvas.configure(scrollregion=canvas.bbox("all"),width=573,height=300))

canvas.create_window((0, 0), window=chatwindow, anchor="nw")

canvas.configure(yscrollcommand=scrollbar.set)

frame.pack()
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

stemmer = LancasterStemmer()
def readData(fileName):
    path = pandas.read_excel(r'D:\Work\University\Sixth Semester\AI\project\final\Virtual real Estate agent\sorted.xlsx', fileName)
    data = path.to_dict('records')
    return data

# main
Datafile = open('intents.json')
data = json.load(Datafile)

# To retrain Models set this to 0
retrain=1

try:
    dividing=10/retrain
    file = open("data.pickle", "rb")
    AllWordsInPatterns, Tags, training, outputs = pickle.load(file)
    file.close()
except:
    AllWordsInPatterns = []
    Tags = []
    Patterns = {}

    for intent in data["intents"]:
        temp = []
        for pattern in intent["patterns"]:

            words = nltk.word_tokenize(pattern)
            AllWordsInPatterns.extend(words)
            temp.append(words)
        Patterns[intent["tag"]] = temp
        if intent["tag"] not in Tags:
            Tags.append(intent["tag"])
    AllWordsInPatterns = [stemmer.stem(w.lower()) for w in AllWordsInPatterns if w not in ["?", "."]]
    AllWordsInPatterns = sorted(list(set(AllWordsInPatterns)))
    training = []
    outputs = []
    for key in Patterns.keys():
        for x, Pattern in enumerate(Patterns[key]):
            bag = []
            words = [stemmer.stem(w.lower()) for w in Pattern]
            for w in AllWordsInPatterns:
                if w in words:
                    bag.append(1)
                else:
                    bag.append(0)
            output = [0 for _ in range(len(Tags))]
            output[Tags.index(key)] = 1

            training.append(bag)
            outputs.append(output)

    training = numpy.array(training)
    outputs = numpy.array(outputs)
    file = open("data.pickle", "wb")
    pickle.dump((AllWordsInPatterns, Tags, training, outputs), file)
    file.close()



tf.compat.v1.reset_default_graph()
try:
    dividing=10/retrain
    model = keras.models.load_model('model.chatbot')
except:
    model = keras.Sequential()
    model.add(keras.layers.Dense(20, activation='relu', input_shape=(len(training[0]),)))
    model.add(keras.layers.Dense(20, activation="relu"))
    model.add(keras.layers.Dense(len(outputs[0]), activation="softmax"))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(training, outputs, epochs=1100, batch_size=8)
    model.save("model.chatbot")

Datafile2 = open('Cities.json')
data2 = json.load(Datafile2)

try:
    dividing=10/retrain
    file = open("data1.pickle", "rb")
    AllWordsInPatternsforCities, TagsforCities, trainingforCities, outputsforCities = pickle.load(file)
    file.close()
except:
    AllWordsInPatternsforCities = []
    TagsforCities = []
    PatternsforCities = {}

    for intent in data2["Cities"]:
        temp = []
        for pattern in intent["patterns"]:

            words = nltk.word_tokenize(pattern)
            AllWordsInPatternsforCities.extend(words)
            temp.append(words)
        PatternsforCities[intent["tag"]] = temp
        if intent["tag"] not in TagsforCities:
            TagsforCities.append(intent["tag"])
    AllWordsInPatternsforCities = [stemmer.stem(w.lower()) for w in AllWordsInPatternsforCities if w not in ["?", "."]]
    AllWordsInPatternsforCities = sorted(list(set(AllWordsInPatternsforCities)))
    trainingforCities = []
    outputsforCities = []
    for key in PatternsforCities.keys():
        for x, Pattern in enumerate(PatternsforCities[key]):
            bag = []
            words = [stemmer.stem(w.lower()) for w in Pattern]
            for w in AllWordsInPatternsforCities:
                if w in words:
                    bag.append(1)
                else:
                    bag.append(0)
            outputforCities = [0 for _ in range(len(TagsforCities))]
            outputforCities[TagsforCities.index(key)] = 1
            trainingforCities.append(bag)
            outputsforCities.append(outputforCities)

    trainingforCities = numpy.array(trainingforCities)
    outputsforCities = numpy.array(outputsforCities)
    file = open("data1.pickle", "wb")
    pickle.dump((AllWordsInPatternsforCities, TagsforCities, trainingforCities, outputsforCities), file)
    file.close()



tf.compat.v1.reset_default_graph()
try:
    dividing=10/retrain
    Citymodel = keras.models.load_model('model.cities')
except:
    Citymodel = keras.Sequential()
    Citymodel.add(keras.layers.Dense(20, activation='relu', input_shape=(len(trainingforCities[0]),)))
    Citymodel.add(keras.layers.Dense(20, activation="relu"))
    Citymodel.add(keras.layers.Dense(len(outputsforCities[0]), activation="softmax"))
    Citymodel.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    Citymodel.fit(trainingforCities, outputsforCities, epochs=1100, batch_size=8)
    Citymodel.save("model.cities")

Datafile3 = open('Areas.json')
data3 = json.load(Datafile3)


try:
    dividing=10/retrain
    file = open("data2.pickle", "rb")
    AllWordsInPatternsforAreas, TagsforAreas, trainingforAreas, outputsforAreas = pickle.load(file)
    file.close()
except:
    AllWordsInPatternsforAreas = []
    TagsforAreas = []
    PatternsforAreas = {}
    for intent in data3["Areas"]:
        temp = []
        for pattern in intent["patterns"]:

            words = nltk.word_tokenize(pattern)
            AllWordsInPatternsforAreas.extend(words)
            temp.append(words)
        PatternsforAreas[intent["tag"]] = temp
        if intent["tag"] not in TagsforAreas:
            TagsforAreas.append(intent["tag"])
    AllWordsInPatternsforAreas = [stemmer.stem(w.lower()) for w in AllWordsInPatternsforAreas if w not in ["?", "."]]
    AllWordsInPatternsforAreas = sorted(list(set(AllWordsInPatternsforAreas)))
    trainingforAreas = []
    outputsforAreas = []
    for key in PatternsforAreas.keys():
        for x, Pattern in enumerate(PatternsforAreas[key]):
            bag = []
            words = [stemmer.stem(w.lower()) for w in Pattern]
            for w in AllWordsInPatternsforAreas:
                if w in words:
                    bag.append(1)
                else:
                    bag.append(0)
            output = [0 for _ in range(len(TagsforAreas))]
            output[TagsforAreas.index(key)] = 1

            trainingforAreas.append(bag)
            outputsforAreas.append(output)

    trainingforAreas = numpy.array(trainingforAreas)
    outputsforAreas = numpy.array(outputsforAreas)
    file = open("data2.pickle", "wb")
    pickle.dump((AllWordsInPatternsforAreas, TagsforAreas, trainingforAreas, outputsforAreas), file)
    file.close()

tf.compat.v1.reset_default_graph()
try:
    dividing=10/retrain
    Areamodel = keras.models.load_model('model.Areas')
except:
    Areamodel = keras.Sequential()
    Areamodel.add(keras.layers.Dense(20, activation='relu', input_shape=(len(trainingforAreas[0]),)))
    Areamodel.add(keras.layers.Dense(20, activation="relu"))
    Areamodel.add(keras.layers.Dense(len(outputsforAreas[0]), activation="softmax"))
    Areamodel.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    Areamodel.fit(trainingforAreas, outputsforAreas, epochs=1100, batch_size=8)
    Areamodel.save("model.Areas")


Datafile4 = open('Prices.json')
data4 = json.load(Datafile4)


try:
    dividing=10/retrain
    file = open("data3.pickle", "rb")
    AllWordsInPatternsforPrices, TagsforPrices, trainingforPrices, outputsforPrices = pickle.load(file)
    file.close()
except:
    AllWordsInPatternsforPrices = []
    TagsforPrices = []
    PatternsforPrices = {}
    for intent in data4["Prices"]:
        temp = []
        for pattern in intent["patterns"]:

            words = nltk.word_tokenize(pattern)
            AllWordsInPatternsforPrices.extend(words)
            temp.append(words)
        PatternsforPrices[intent["tag"]] = temp
        if intent["tag"] not in TagsforPrices:
            TagsforPrices.append(intent["tag"])
    AllWordsInPatternsforPrices = [stemmer.stem(w.lower()) for w in AllWordsInPatternsforPrices if w not in ["?", "."]]
    AllWordsInPatternsforPrices = sorted(list(set(AllWordsInPatternsforPrices)))
    trainingforPrices = []
    outputsforPrices = []
    for key in PatternsforPrices.keys():
        for x, Pattern in enumerate(PatternsforPrices[key]):
            bag = []
            words = [stemmer.stem(w.lower()) for w in Pattern]
            for w in AllWordsInPatternsforPrices:
                if w in words:
                    bag.append(1)
                else:
                    bag.append(0)
            outputforPrices = [0 for _ in range(len(TagsforPrices))]
            outputforPrices[TagsforPrices.index(key)] = 1

            trainingforPrices.append(bag)
            outputsforPrices.append(outputforPrices)

    trainingforPrices = numpy.array(trainingforPrices)
    outputsforPrices = numpy.array(outputsforPrices)
    file = open("data3.pickle", "wb")
    pickle.dump((AllWordsInPatternsforPrices, TagsforPrices, trainingforPrices, outputsforPrices), file)
    file.close()

tf.compat.v1.reset_default_graph()
try:
    dividing=10/retrain
    Pricemodel = keras.models.load_model('model.Prices')
except:
    Pricemodel = keras.Sequential()
    Pricemodel.add(keras.layers.Dense(20, activation='relu', input_shape=(len(trainingforPrices[0]),)))
    Pricemodel.add(keras.layers.Dense(20, activation="relu"))
    Pricemodel.add(keras.layers.Dense(len(outputsforPrices[0]), activation="softmax"))
    Pricemodel.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    Pricemodel.fit(trainingforPrices, outputsforPrices, epochs=1100, batch_size=8)
    Pricemodel.save("model.Prices")


def bag_of_words(Input, AllWords):
    bag = [0 for _ in range(len(training[0]))]
    words = nltk.word_tokenize(Input)
    words = [stemmer.stem(word.lower()) for word in words]
    for word in words:
        for i, w in enumerate(AllWords):
            if w == word:
                bag[i] = 1
    return numpy.array(bag)

def bag_of_wordsforCities(Input, AllWords):
    bag = [0 for _ in range(len(trainingforCities[0]))]
    words = nltk.word_tokenize(Input)
    words = [stemmer.stem(word.lower()) for word in words]
    for word in words:
        for i, w in enumerate(AllWords):
            if w == word:
                bag[i] = 1
    return numpy.array(bag)

def bag_of_wordsforAreas(Input, AllWords):
    bag = [0 for _ in range(len(trainingforAreas[0]))]
    words = nltk.word_tokenize(Input)
    words = [stemmer.stem(word.lower()) for word in words]
    for word in words:
        for i, w in enumerate(AllWords):
            if w == word:
                bag[i] = 1
    return numpy.array(bag)

def bag_of_wordsforPrices(Input, AllWords):
    bag = [0 for _ in range(len(trainingforPrices[0]))]
    words = nltk.word_tokenize(Input)
    words = [stemmer.stem(word.lower()) for word in words]
    for word in words:
        for i, w in enumerate(AllWords):
            if w == word:
                bag[i] = 1
    return numpy.array(bag)

rows = 2
list = []
index1=0
flags=[0,0,0,0]
Information=[0,0,0,0,0,0]
def GetResponce():
    if len(Information) == 0:
        response = " It is sad that the information you gave is not enough to suggest the best. Could you please give more details of what you want to buy?"
    if flags[0]==0:
        response = "Please tell me would you like to buy a house or a flat"
    elif flags[1] == 0:
        response = "I think I like you. Can you tell me what city would you prefer for property?"
    elif flags[2] == 0:
        if flags[3] == 0:
            response = "Cool! You can mention the Area in 'Marla' or 'Kannal'. The price in which you would like to buy. If you like I could start giving you suggections"
        else:
            response = "Cool! You can mention the Area in 'Marla' or 'Kannal' of which you would like to buy or You can ask me to suggest right away!"
    elif flags[3] == 0:
        response = "You can mention the price in which you would like to buy or should i just start telling you suggestions"
    if flags[0]==1 and flags[1]==1 and flags[2]==1 and flags[3]==1:
        response="Thank you for providing with all the information needed you can now begin asking me for options and i would try my best to satisfy you"
    return response
def chat():
    def keyup():
        global flags
        User_Input = e1.get()
        print(User_Input)
        if User_Input.lower() == "quit":
            exit()

        results = model.predict([[bag_of_words(User_Input, AllWordsInPatterns)]])
        result_index = numpy.argmax(results)
        if results[0][result_index] > 0.8:
            for tag in data["intents"]:
                if tag['tag'] == Tags[result_index]:
                    if "Suggest" == Tags[result_index]:
                        global index1
                        if flags[0]==1 and flags[1]==1:
                            data1 = readData("Location")
                            list.clear()
                            for h in data1:
                                if flags[3]==1 and flags[2]==1:
                                    if h["type"]==Information[0] and h["City"]==Information[1] and h["Area"]==Information[2] and h["price"]==Information[3]:
                                        list.append(h)
                                elif flags[2]==1:
                                    if h["type"]==Information[0] and h["City"]==Information[1] and h["Area"]==Information[2]:
                                        list.append(h)
                                elif flags[3]==1:
                                    if h["type"]==Information[0] and h["City"]==Information[1] and h["price"]==Information[3]:
                                        list.append(h)
                                else:
                                    if h["type"]==Information[0] and h["City"]==Information[1]:
                                        list.append(h)
                            if len(list)>index1:
                                string =str("City:"+list[index1]['City']+", Landmark:"+list[index1]['Landmark']+", Type:"+list[index1]['type']+", Price:"+list[index1]['price']+", Location:"+list[index1]['location']+", Area:"+list[index1]['Area'])
                                string+="\n Would you like me to suggest another?"
                                index1 = index1 + 1
                            else:
                                list.clear()
                                index1=0
                                flags=[0,0,0,0]
                                string="OOPS! I couldn't find any property that met your needs. Can you try with different requirements?"
                            response=string
                        else:
                            reponse= GetResponce()
                    else:
                        if "House" == Tags[result_index]:
                            flags[0] = 1
                            Information[0]="House"
                            response = GetResponce()
                        elif "Flat" == Tags[result_index]:
                            flags[0] = 1
                            Information[0]="Flat"
                            response=GetResponce()
                        else:
                            responses = tag['responses']
                            response=random.choice(responses)
                        break
            T = Label(chatwindow, text=response, width=40, bg="grey", wraplength=280)
        else:
            count=0
            results = Citymodel.predict([[bag_of_wordsforCities(User_Input, AllWordsInPatternsforCities)]])
            result_index = numpy.argmax(results)
            if results[0][result_index] > 0.8:
                for tag in data2["Cities"]:
                    if tag['tag'] == TagsforCities[result_index]:
                        count=count+1
                        flags[1] = 1
                        Information[1]=tag['tag']

            results = Areamodel.predict([[bag_of_wordsforAreas(User_Input, AllWordsInPatternsforAreas)]])
            result_index = numpy.argmax(results)
            if results[0][result_index] > 0.8:
                for tag in data3["Areas"]:
                    if tag['tag'] == TagsforAreas[result_index]:
                        count = count + 1
                        flags[2] = 1
                        Information[2]=tag['tag']

            results = Pricemodel.predict([[bag_of_wordsforPrices(User_Input, AllWordsInPatternsforPrices)]])
            result_index = numpy.argmax(results)
            print(result_index)
            if results[0][result_index] > 0.8:
                for tag in data4["Prices"]:
                    if tag['tag'] == TagsforPrices[result_index]:
                        count = count + 1
                        flags[3] = 1
                        Information.insert(3, tag["tag"])

            if count ==0:
                T = Label(chatwindow, text="Sorry, I could'nt get you. Can you elaborate?", width=40,bg="grey",wraplength=270)
            else:
                T = Label(chatwindow, text=GetResponce(), width=40, bg="grey", wraplength=270)

        
        global rows 
        Label1 = Label(chatwindow, text=User_Input, width=40,bg="green")
        Label1.grid(row=rows-1,column=0)
        T.grid(row=rows, column=1)
        rows = rows+2
        e1.delete(first=0, last=200)

    frame1=Frame(root,bg="black")
    e1 = Entry(frame1,width=95,bd=5)
    e1.grid(row=0, column=0,columnspan=2)
    e1.bind("<Return>",lambda event: keyup())
    frame1.pack()
    root.mainloop()

chat()
Datafile.close()




"""
df = readData("Price")
temp=""
id=1
for index, row in df.iterrows():
    if index==0:
        dist = {"tag": "Price "+str(id),
                "patterns": [str(row['price'])],
                "responses": ["I have all the essential information i need would you like me to suggest a few options"],
                "context_set": ""
                }
        id=id+1
        temp=row['price']
    if temp!=row['price']:
        data["intents"].append(dist)
        dist = {"tag": "Price "+str(id),
                "patterns": [row['price']],
                "responses": ["I have all the essential information i need would you like me to suggest a few options"],
                "context_set": ""
                }
        id=id+1
        temp = row['price']
data["intents"].append(dist)
json_object = json.dumps(data,indent=5)

with open("intents.json", "w") as outfile:
    outfile.write(json_object)

df = readData("Area")
temp=""
id=1
for index, row in df.iterrows():
    if index==0:
        dist = {"tag": "Area "+str(id),
                "patterns": [str(row['Area'])],
                "responses": ["What is the price you are willing to buy at"],
                "context_set": ""
                }
        id=id+1
        temp=row['price']
    if temp!=row['price']:
        data["intents"].append(dist)
        dist = {"tag": "Price "+str(id),
                "patterns": [row['Area']],
                "responses": ["What is the price you are willing to buy at"],
                "context_set": ""
                }
        id=id+1
        temp = row['price']
data["intents"].append(dist)
json_object = json.dumps(data,indent=5)

with open("intents.json", "w") as outfile:
    outfile.write(json_object)

df = readData("Location")
temp=""
id=1
for index, row in df.iterrows():
    if index==0:
        dist = {"tag": "City "+str(id),
                "patterns": [str(row['City'])],
                "responses": ["What size(marla or kanal) of property you are looking for"],
                "context_set": ""
                }
        id=id+1
        temp=row['City']
    if temp!=row['City']:
        data["intents"].append(dist)
        dist = {"tag": "City "+str(id),
                "patterns": [row['City']],
                "responses": ["What size(marla or kanal) of property you are looking for"],
                "context_set": ""
                }
        id=id+1
        temp = row['City']
data["intents"].append(dist)
json_object = json.dumps(data,indent=5)

with open("intents.json", "w") as outfile:
    outfile.write(json_object)

"""
