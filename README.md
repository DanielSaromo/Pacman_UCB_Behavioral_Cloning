# Pacman_UCB_Behavioral_Cloning
Entorno de Pacman UCB preparado para hacer Behavioral Cloning.

Fuente original del entorno de juego: https://inst.eecs.berkeley.edu/~cs188/sp21/project2/.

Modifiqué dichos códigos para que puedan ser utilizados por una IA que utiliza Behavioral Cloning.
Requiere un cuadernillo de extensión `ipynb` para entrenar el modelo de ML supervisado que controlará al agente de Pacman.

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/DanielSaromo/Pacman_UCB_Behavioral_Cloning)

**_Actualización_:** Se han agregado los archivos `playGymEnv.py` y `ddqn_sol.py`, que permiten ejecutar y visualizar un agente DDQN (Double Deep Q-Network) en un entorno de la librería OpenAI Gym. Están soportados dos agentes, uno para el entorno vanilla, y otro para el entorno augmentado con teoría vista en clase. Los modelos neuronales de los agentes son entrenados externamente en un cuadernillo de extensión `ipynb`. Una gráfica que muestra el resultado de dos agentes entrenados, se puede encontrar en el archivo `ddqn_twoEnvs.pdf`.

---

- Repositorio elaborado por: Daniel Saromo Mori (Contacto: www.danielsaromo.xyz).
- Curso: Inteligencia Artificial Para Juegos.
- El curso pertenece a la **Diplomatura de Especialización en Desarrollo de Aplicaciones con Inteligencia Artificial**, brindada por la **Pontificia Universidad Católica del Perú (PUCP)**.