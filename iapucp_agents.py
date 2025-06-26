from game import Directions
from game import Agent
from game import Actions

import random
import util
import searchAgents

AGENTS_SPEED = 1.0 # Este valor puede ser 0.5 o 1

# Las jugadas por defecto que están grabadas en la carpeta replayFolder, tienen un AGENTS_SPEED de 1.0

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
import tensorflow as tf
import tensorflow.keras as keras
print('tensorflow: {}'.format(tf.__version__))
# pickle
import pickle
print('pickle: {}'.format(pickle.format_version))

# Fin de importación de librerías

# no olvidar actualizar la variable N!
N=4 # el vector de atributos extra. para el código de ejemplo, cada columna tiene 2 valores posibles
cantFeatures = 10 + N

# FUNCIÓN PARA OBTENER FEATURES A PARTIR DE UN GAMESTATE
def obtenerFeatures(gState, agregarExtraAttributes=True):
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
    
    # Los siguientes códigos permiten corregir la escala, según el valor de AGENTS_SPEED usado
    raw_pac = gState.getPacmanPosition()
    factor = 1.0 / AGENTS_SPEED

    # properly round to the nearest integer grid‐cell
    px = ( raw_pac[0] / AGENTS_SPEED + factor/2 ) // factor
    py = ( raw_pac[1] / AGENTS_SPEED + factor/2 ) // factor
    pac_pos = ( int(px), int(py) )

    ghosts_raw = gState.getGhostPositions()
    ghosts_poss = []
    for gx, gy in ghosts_raw:
        gx_sc = ( gx / AGENTS_SPEED + factor/2 ) // factor
        gy_sc = ( gy / AGENTS_SPEED + factor/2 ) // factor
        ghosts_poss.append(( int(gx_sc), int(gy_sc) ))

    ghosts_poss_relToPacman = np.array([
        np.array(g) - np.array(pac_pos)
        for g in ghosts_poss
    ], dtype=int)

    #print("pacposs", pac_pos)
    #print("fantasmiposs", ghosts_poss)

    #print("relativePos", ghosts_poss_relToPacman)

    ############## ############## Features ############## ##############

    # Feature de posición de fantasmas respecto a pacman
    # 1. 'posx_g1_fromPacman' ('g1x')
    # 2. 'posy_g1_fromPacman' ('g1y') 
    # 3. 'posx_g2_fromPacman' ('g2x')
    # 4. 'posy_g2_fromPacman' ('g2y')
    features = np.append(features, ghosts_poss_relToPacman)
    
    # Feature de cantidad de capsulas
    # 5. 'capsulas_restantes' ('caps')
    capsules = gState.getCapsules()
    features = np.append(features, len(capsules))
    
    # Configuración de food
    state_food = gState.getFood()
    food = [(x, y) #enlista las posiciones donde hay comida
            for x, row in enumerate(state_food)
            for y, food in enumerate(row)
            if food]
 
    # Feature de Fantasmita Mas Cercano: a cuanta distancia manhattan esta el fantasma mas cercano
    # 6. 'manhattan_dist_toClosest_ghost' ('dist_ghost')
    nearest_ghosts = sorted([util.manhattanDistance(pac_pos, i) for i in ghosts_poss])
    features = np.append(features, [ nearest_ghosts[0] ])
    

    # Feature de Pildora mas cercana #a cuanta distancia manhattan esta la capsula mas cercana
    # 7. 'manhattan_dist_toClosest_caps' ('dist_caps')
    nearest_caps = sorted([util.manhattanDistance(pac_pos, i) for i in capsules])
    if nearest_caps:
        manhDist_nearestCaps = nearest_caps[0]
    else:
        manhDist_nearestCaps = max(gState.data.layout.width,gState.data.layout.height)
    features = np.append(features, [manhDist_nearestCaps])
    
    # Feature del promedio de MD a las 5 comidas mas cercanas. Que pasa cuando hay menos de 5?
    # 8. 'mean_manhDist_5_closests_dots' ('dist_5dots')
    nearest_food = sorted([(util.manhattanDistance(pac_pos, i),i) for i in food])
    nearest_food = nearest_food[:5]
    for i in range(min(len(nearest_food), 5)):
        nearest_food[i]=searchAgents.mazeDistance(pac_pos,nearest_food[i][1],gState)

    features = np.append(features, sum(nearest_food)/len(nearest_food))

    # Feature de Score
    # 9. 'score'
    features = np.append(features, [gState.getScore()] )

    # Feature de cantidad de Fantasmitas Asustaditos
    # 10. 'cant_scared_ghosts' ('scared')
    ghostStates = gState.getGhostStates()
    numOfScaredGhosts = 0
    for ghostState in ghostStates:
        if ghostState.scaredTimer > 0:
            numOfScaredGhosts += 1

    # este es el vector con los 10 atributos iniciales
    features = np.append(features, [numOfScaredGhosts] )

    ####### Agregamos un vector N-dimensional con los N atributos extra
    
    # Leyenda de los N=4 elementos que vamos a agregar
    # en el código de ejemplo se agregarán N=4 elementos
    # (agregaremos un elemento por cada una de las 4 acciones fisicas de mov.)
    # 1: no wall (dirección válida para moverse)
    # 2: wall

    # Así se obtiene la lista con las acciones legales, a partir del state actual
    legalActions = gState.getLegalActions()

    #print("Acciones legales:", legalActions)

    lista_newAttributes = []
    lista_all_relevant_moves = ['West', 'East', 'North', 'South']

    for ii in range(0,4): # N=4, ya que en este ejemplo agregamos una dimension por cada dirección de mov.
        # si no es una acción legal, es porque en esa dirección hay un wall
        if lista_all_relevant_moves[ii] not in legalActions:
            feature_movil = 2
        else:
            feature_movil = 1 

        lista_newAttributes.append(feature_movil)

    # Aquí agregamos los N=4 atributos extra
    if agregarExtraAttributes: features = np.append(features, lista_newAttributes)
    
    #print(features) # en este código de ejemplo, el único valor decimal (non integer) es el
    # que corresponde al Promedio de las dists manhattan de las 5 comidas más cercanas

    return features

# CLASE QUE IMPLEMENTA UN AGENTE ALEATORIO PERSONALIZABLE
class my_Random_Agent(Agent):
    """
    This is a RANDOM agent!
    """

    def __init__(self):

        print()
        print("="*15)
        print()
        print("Se inicializó el agente ALEATORIO")
        print("="*15)
   

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        features = obtenerFeatures(state).reshape(1,-1)

        # Así se obtiene la lista con las acciones legales, a partir del state actual
        legalActions = state.getLegalActions()

        return legalActions[  random.randint(1,len(legalActions))-1 ]

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

        #Si es un DecisionTreeClassifier o un RandomForestClassifier
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