from game import Directions
from game import Agent
from game import Actions

import random

import util
import searchAgents

#Importando librerías para ML
# requiere haber instalado (sugerencia: usar pip): scipy, numpy, matplotlib, pandas, 
# sklearn, keras, tensorflow

# import warning en tf: https://stackoverflow.com/questions/66092421/how-to-rebuild-tensorflow-with-the-compiler-flags
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# version de Python
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy as np
print('numpy: {}'.format(np.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas as pd
print('pandas: {}'.format(pd.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
# keras
import tensorflow.keras as keras
print('keras: {}'.format(keras.__version__))
# pickle
import pickle
print('pickle: {}'.format(pickle.format_version))

# Fin de importación de librerías

# FUNCIÓN PARA OBTENER FEATURES A PARTIR DE UN GAMESTATE
def obtenerFeatures(gState):
    """
    Esta es la función que obtiene los features a partir de un GameState del entorno.
    Para el desafío, no es necesario modificar esta función, pero si desea, puede hacerlo, para obtener mejores features para el aprendizaje.
    Características extraidas [12]: Posicion en x del ghost1 respecto
    a pacman, Posicion en y del ghost1 respecto a pacman, Posicion en x del ghost2 respecto a pacman,
    Posicion en y del ghost2 respecto a pacman, Cantidad de cápsulas restantes,
    Distancia manhattan al fantasmita más cercano, Distancia manhattan a la cápsula más cercana,
    Promedio de las distancias manhattan de las 5 comidas más cercanas (no capsulas), Score,
    Cantidad de fantasmas asustados
    """
    features = np.array([])
    #gState_successor = gState.generateSuccessor(0, accion)
    #isWin (1/0), isLose (1/0)
    #features = np.append(features, [ int(gState_successor.isWin()) , int(gState_successor.isLose()) ])

    pac_pos = gState.getPacmanPosition()
    ghosts_poss = gState.getGhostPositions()

    ghosts_poss_relToPacman = np.array([np.array(x) - np.array(pac_pos) for x in ghosts_poss])

    features = np.append(features, ghosts_poss_relToPacman)
    
    capsules = gState.getCapsules()

    # Feature de cantidad de capsulas
    features = np.append(features, len(capsules))

    state_food = gState.getFood()
    food = [(x, y) #enlista las posiciones donde hay comida
            for x, row in enumerate(state_food)
            for y, food in enumerate(row)
            if food]
    nearest_ghosts = sorted([util.manhattanDistance(pac_pos, i) for i in ghosts_poss])

    # Feature de Fantasmita Mas Cercano: a cuanta distancia manhattan esta el fantasma mas cercano
    features = np.append(features, [ nearest_ghosts[0] ])
    ############################lo de arriba esta bien
    # Feature de Pildora mas cercana #a cuanta distancia manhattan esta la capsula mas cercana
    nearest_caps = sorted([util.manhattanDistance(pac_pos, i) for i in capsules])
    if nearest_caps:
        manhDist_nearestCaps = nearest_caps[0]
    else:
        manhDist_nearestCaps = max(gState.data.layout.width,gState.data.layout.height)
    features = np.append(features, [manhDist_nearestCaps])
    # Feature del promedio de MD a las 5 comidas mas cercanas. Que pasa cuando hay menos de 5?
    nearest_food = sorted([(util.manhattanDistance(pac_pos, i),i) for i in food])
    nearest_food = nearest_food[:5]
    for i in range(min(len(nearest_food), 5)):
        nearest_food[i]=searchAgents.mazeDistance(pac_pos,nearest_food[i][1],gState)

    features = np.append(features, sum(nearest_food)/len(nearest_food))

    # Feature de Score
    features = np.append(features, [gState.getScore()] )

    # Feature de cantidad de Fantasmitas Asustaditos
    ghostStates = gState.getGhostStates()
    numOfScaredGhosts = 0
    for ghostState in ghostStates:
        if ghostState.scaredTimer > 0:
            numOfScaredGhosts += 1
    features = np.append(features, [numOfScaredGhosts] )

    #mazeDistanceAndFirstAction
    return features.astype(int)

# CLASE QUE IMPLEMENTA EL AGENTE BASADO EN BEHAVIORAL CLONING
class my_ML_Agent(Agent):
    """
    This is a behaviour clonned agent!
    """

    def __init__(self):
        import pickle
        import numpy as np

        # open a file, where you stored the pickled data
        file_modeloCargado = open('modeloEntrenado.p', 'rb')

        # load information from that file
        self.modelo = pickle.load(file_modeloCargado)

        # close the file
        file_modeloCargado.close()

        self.cantAccionesInvalidas = 0

        print()
        print("="*15)
        print()
        print("Se inicializó el agente basado en Behavior Cloning")
        print("Autor: Grupo número (?)\n")
        print("="*15)
   

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        features = obtenerFeatures(state).reshape(1,-1)

        #Si es un DecisionTreeClassifier
        accionNum = self.modelo.predict(features)

        #Si es un keras sequential
        #accionNum = self.modelo.predict(features).argmax(axis=-1)

        #Convertir el índice de la acción a su respectivo string
        # TO DO

        # Así se obtiene la lista con las acciones legales, a partir del state actual
        legalActions = state.getLegalActions()

        ####Si deseas, usa la variable `self.cantAccionesInvalidas`

        #Codificar el comportamiento del agente para el caso de predecir una accion inválida
        # TO DO

        return 'East'