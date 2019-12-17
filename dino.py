#import sys
#sys.setrecursionlimit(15000)

import pyautogui
import time
import numpy as np
import os
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Activation,Input
from numpy import array
import pandas as pd

from sklearn.model_selection import train_test_split
RGBcolor = [83,83,83]

def open_game(): #ouvrir le jeu dans une fenétre chromium
	os.system("chromium-browser trex/index.html  --new-window &")


def load_game(mode): #selectionner la fenétre du jeu (dans le cas ou elle ne serait pas la fenétre active)
	os.system("xdotool search 'T-rex game'>temporary_file")
	temporary_file=open("temporary_file")
	window_id=temporary_file.readline()
	if len(window_id) < 2:
		print("No window detected!")
		return False
	else:
		os.system("xdotool windowactivate "+str(window_id))
		return True


def detect_dinosor(): #détécter le player 
	dinosor_position = pyautogui.locateOnScreen('dinosor.png')
	if not dinosor_position:
		print("No dinosor")
		return False
	else:
		fromx=int(dinosor_position[0])+72
		fromy=int(dinosor_position[1])-30
		print(dinosor_position,' AND X:',fromx,' AND Y:',fromy)
		return fromx,fromy



def record(fromx,fromy):  #mesure de la distance entre dinosor et l'obstact
	
	screenShot = np.array(pyautogui.screenshot(region=(fromx,fromy, 800, 32)))
	'''
	screenShot1 = pyautogui.screenshot(region=(fromx+10,fromy, 800, 32))
	screenShot1.save("img.png")
	screenShot=np.array(screenShot1)
	'''
	for x in range(0,800):
		for y in range(7,18):
			if screenShot[y][x][0] == RGBcolor[0] & screenShot[y][x][1] == RGBcolor[1] & screenShot[y][x][2] == RGBcolor[2]:
				print(x)
				return x
	return 800







def jump(): #fonction pour que le player saute
	os.system("xdotool type  ' ' ")
	

def game_over(fromx,fromy): #fonction detecter la fin de la partie 
	gameover = pyautogui.locateOnScreen('game_over.png', region=(fromx+190,fromy-20, 80, 55))
	if not gameover:
		return False
	else:
		return True

def creat_first_model(model_path): #le modéle 
	inputs = Input(shape=(2,))
	x = Dense(2, activation='relu')(inputs)
	x = Dense(7, activation='relu')(x)
	predictions = Dense(1, activation='sigmoid')(x)
	model = Model(inputs=inputs, outputs=[predictions])
	model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
	print (model.summary())
	model.save(model_path)

	
def predict(distance,jump): 

	X = array([[distance,jump]]) #fonction traduisant la prédiction en booléen 
	if loaded_model.predict(X) > 0.49:
		return True
	else:
		return False 

def saveToDataset(distance,jump,alive): #fonction sauvgardant les données dans un dataset
	if distance!=0:
		log.write(str(distance)+','+str(jump)+','+str(alive)+'\n')


def fit_data(model_path): #entrainement du modéle a l'aide du dataset généré 
	pdata= pd.read_csv('Log.csv', sep=',')
	Labels=pdata['Alive']
	print("Labels:\n",Labels,"\nEND")
	features = pdata[['Distance','Jump']]
	X = features
	y = Labels
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
	inputs = Input(shape=(2,))
	x = Dense(2, activation='relu')(inputs)
	x = Dense(7, activation='relu')(x)
	predictions = Dense(1, activation='sigmoid')(x)
	model = Model(inputs=inputs, outputs=[predictions])
	model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
	print (model.summary())
	model.fit(X_train, y_train, epochs=250, batch_size=10,verbose=1)  # starts training
	scores = model.evaluate(X_test.values, y_test, verbose=2)
	print("\n accuracy:::%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	model.save(model_path)
	return scores[1]*100
	

def play(x,y):  # fonction définisant l'action a entreprendre en fonction de la prédiction du modéle
	lives=20
	#lives=500 
	newdistance=0
	jump()
	while lives>1:
		alive=1
		lastdistance=newdistance   # enregistrement de la derniére distance mesuré
		newdistance=record(x,y)
		if predict(newdistance,1):  # action en fonction de la prédiction 
			jump()
			time.sleep(0.5)
		else:
			time.sleep(0.5)
		if game_over(x,y):
			lives-=1
			alive=0
			print("Game Over!!")
			print("Number of Lives left:",lives)
			jump()
		saveToDataset(lastdistance,1,alive)
		



creat_first_model("Model0.h5") #créer le premier modéle (non entrainé)
loaded_model = load_model("Model0.h5")
loaded_model.summary()
log=open("Log.csv","w+")      #création du dataset si non existant 
log.write("Distance,Jump,Alive\n")
log.close
log=open("Log.csv","a+")   #enregistrer les données dans le dataset "log.csv"
open_game()					# ouvrir la fenétre du jeu
time.sleep(5)
if load_game(1):			#démarrer le jeu 
	time.sleep(2)
	if detect_dinosor() != False:  
		x0,y0 = detect_dinosor()	#enregistrer la position du dinosor

		play(x0,y0)   #jouer 
		log.close()
		accuracy=fit_data("TrainedModel.h5")  #entrainer le modéle
		log=open("Log.csv","a+")
		while accuracy < 95:                   #répéter l'opération tant que la précision est inférieur a 95%
			loaded_model = load_model("TrainedModel.h5")
			loaded_model.summary()
			play(x0,y0)
			accuracy=fit_data("TrainedModel.h5")
		print("Termine! precision > a 95%")       #modéle finale entrainé obtenue!