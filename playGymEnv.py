# -*- coding: utf-8 -*-
# Códigos para probar Agentes DDQN en Entornos OpenAI Gym
# Elaborado por: Daniel Saromo Mori (Contacto: www.danielsaromo.xyz)
# Curso: Inteligencia Artificial Para Juegos
# El curso pertenece a la Diplomatura de Especialización en Desarrollo de Aplicaciones con Inteligencia Artificial,
# brindada por la Pontificia Universidad Católica del Perú (PUCP)

# Instrucciones:
# 1) Subir dos archivos con las redes neuronales de sus agentes ya entrenados en su cuadernillo ipynb:
#    `agent.keras`, y `agent_v2.keras`.
# 2) En el archivo `ddqn_sol.py`, debe completar la implementación de:
#    `EstimadorEstadosInternos`, y `get_reward`. Seguir la teoría vista en clase. Deben ser
#    las mismas implementaciones usadas en su cuadernillo ipynb.
# 3) En la línea de comandos, ejecutar: `python playGymEnv.py`.
# 4) NO debe modificar el presente archivo (`playGymEnv.py`).

# Nota: Si no se modifica la función `get_reward`, pero sí se cumplen las demás instrucciones,
# el entorno sí será ejecutado, pero el valor de reward y score_puro obtenidos, serán 0.

from ddqn_sol import EstimadorEstadosInternos, get_reward
from tensorflow.keras.saving import load_model
import tensorflow as tf
import numpy as np
import random
import gym

class DDQNAgent_reconstruido():
    def __init__(self, model_filename='agent.keras'):
        print("Asegurarse que la versión de TF usada para grabar las redes neuronales es 2.15.0")
        self.model = load_model(model_filename) # versión de tf usada para grabar las RNs: 2.15.0
        self.state_size = self.model.input_shape[1]
        self.action_size = self.model.output_shape[1]

def play_MountainCar(agentito, trials = 10):
    env = gym.make('MountainCar-v0')
    agentito.model.compile()
    scores = []
    state_size = agentito.state_size

    print("Cantidad de intentos para el agente:", trials)

    for trial in range(1,trials+1):
        game_memory = []
        state = env.reset()

        estimador = EstimadorEstadosInternos(state)
        state = estimador.calculaEstadosInternos(state) # esta línea permite verificar si el estimador funciona bien
        state = []

        score = 0
        score_puro = 0
        max_pos_reached = -9999
        num_ofExtraFeats = agentito.state_size - env.reset().shape[0]

        if trial == 1: print("Número de augmented features del entorno:", num_ofExtraFeats)     

        for step in range(1500): # en cada trial ejecuta 1500 pasos
            env.render() # para visualizar en el entorno al agente entrenado previamente

            if len(state) == 0: # si es el primer movimiento  -> escoge una accion aleatoria
                action = random.randrange(0,env.action_space.n)
            else:
                action_values = agentito.model.predict(state.reshape(1, -1), verbose=0) # predice los q valores con la RN del agentito
                action = np.argmax(action_values[0]) # retorna la accion con el maximo q-valor predicho

            next_state, reward, done, _  = env.step(action) # corre el entorno un step ejecutando la accion inferida
         
            reward = get_reward(next_state)
            max_pos_reached=max(max_pos_reached, next_state[0])
            # al reward, agregaremos una bonificación extra relacionada a la mayor posición en x alcanzada
            reward += max_pos_reached*10

            if num_ofExtraFeats != 0:
                next_state = estimador.calculaEstadosInternos(next_state)

            next_state = np.reshape(next_state, [1, state_size])
            state = next_state # actualiza el estado actual al nuevo estado

            score += reward # el reward es lo ganado en este step. el score es el acumulado de rewards para el episodio

            # score_puro es el score, pero sin la bonficación extra relacionada a la máxima posición en x alcanzada
            score_puro += (reward-max_pos_reached*10)
            
            game_memory.append([next_state, action])

            if done:
                print("Played episode: {}/{}, score: {:.4}".format(str(trial).zfill(3), trials, score))
                print("max pos reached: {:.4}".format(max_pos_reached))
                print("score_puro:", score_puro)
                break

        scores.append(score)
    env.close()

    print("Score medio = {}".format(sum(scores) /float(trials)) )

if __name__ == '__main__':
    """
    The main function called when playGymEnv.py is run
    from the command line:

    > python playGymEnv.py
    """
    print("Gym version:", gym.__version__) # códigos probados en versión 0.22.0: https://github.com/openai/gym/releases/tag/0.22.0
    print("TF version:", tf.__version__) # códigos probados en versión 2.15.0

    #args = readCommand(sys.argv[1:])  # Get game components based on input
    #runGames(**args)

    assert(gym.__version__=="0.22.0"), "La versión de gym requerida es 0.22.0"
    assert( tf.__version__=="2.15.0"), "La versión de tensorflow requerida es 2.15.0"

    agent_onVanillaEnv = DDQNAgent_reconstruido('agent.keras')
    agent_onAugmentedEnv = DDQNAgent_reconstruido('agent_v2.keras')

    print("-"*100)
    print("--- Agente en Entorno Vanilla ---")
    play_MountainCar(agent_onVanillaEnv, trials=5)

    print("-"*100)
    print("--- Agente en Entorno Augmentado ---")
    play_MountainCar(agent_onAugmentedEnv, trials=5)

    # import cProfile
    # cProfile.run("runGames( **args )")
    pass