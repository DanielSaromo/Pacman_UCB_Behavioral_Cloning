# -*- coding: utf-8 -*-

import numpy as np

# El entorno MountainCar-v0, de gym versión 0.22.0, tiene un estado con 2 elementos:
# posición horizontal y velocidad
# https://gymnasium.farama.org/environments/classic_control/mountain_car/
# https://github.com/openai/gym/wiki/MountainCar-v0
def get_reward(state):
    x=state[0]
    v=state[1]
    if x >= 0.5:
        print("Car has reached the goal 😁")
        #No está permitido editar las anteriores líneas de código de esta función!
    return 0 # Usted debe completar la implementación de esta función

class EstimadorEstadosInternos():
    def __init__(self, initial_state):
        #...
        #...
        pass

    def calculaEstadosInternos(self, nuevo_estado):
        estado_augmentado = np.array([nuevo_estado[0], nuevo_estado[1],
                                #...          
                                      ])
        #...
        #...
        return estado_augmentado