# 🚀 Arquitectura HADRA (Hybrid Autonomous Drone Reinforcement Agents)

## 📝 Descripción

HADRA es una arquitectura híbrida de agentes de IA diseñada para planificar y ejecutar misiones críticas con un enjambre de drones.

### 🏗️ Componentes:
- **👩‍🚀 Agente Madre de Colmena:** Módulo de planificación global basado en un LLM (por ejemplo, Qwen/1.5).
- **🐝 Microagentes Zánganos (HADRA Units):** Cada dron tiene un microagente con:
  - 🧠 **Núcleo Neuro-Simbólico (NS):** Red neuronal ligera + reglas IF-THEN.
  - 🔧 **Núcleo de Meta-Aprendizaje (ML):** Ajusta hiperparámetros en tiempo real.
  - 👀 **Núcleo de Consciencia Operacional (OA):** Monitoreo del estado y entorno.
  - 🤝 **Núcleo de Cooperación Emergente (EC):** Coordinación y formación en grupo.

## ⚙️ Funcionalidades

- **🎭 Entrenamiento Actor-Crítico:** Microagentes aprenden a alcanzar sub-metas asignadas.
- **📡 Planificación Jerárquica:** Agente madre genera sub-metas.
- **📊 Registro y Visualización:**
  - 📝 Métricas de recompensa y trayectorias en CSV.
  - 📺 Dashboard interactivo (Plotly Dash) para analizar misiones en tiempo real.

## 📂 Estructura del Repositorio
```
HADRA_GPT/
├── HADRA/
│   ├── configs/
│   │   ├── airsim.json
│   │   └── training.yaml
│   ├── core/
│   │   ├── agent.py
│   │   ├── env.py
│   │   ├── hierarchical_agent.py
│   │   ├── micro_agent.py
│   │   └── utils.py
│   └── scripts/
│       ├── airsim_launcher.py
│       ├── deploy.py
│       ├── mission_deploy.py
│       ├── monitor.py
│       ├── train.py
│       └── train_mission.py
├── data/
│   ├── pretrained/  # Modelos guardados
│   └── training_log.csv  # Log de entrenamiento
├── logs/
│   └── telemetry_log.csv  # Telemetría de drones
└── README.md
```

## 🛠️ Cómo Ejecutar el Proyecto

1. **🎮 Configurar AirSim:**
   - Asegúrate de tener AirSim corriendo en Unreal Engine 4.27 con el mapa Blocks.
   - Ejecuta:
     ```bash
     python -m HADRA.scripts.airsim_launcher
     ```

2. **🖥️ Monitorear la Simulación:**
   ```bash
   python -m HADRA.scripts.monitor
   ```

3. **📈 Entrenamiento:**
   ```bash
   python -m HADRA.scripts.train_mission --episodes 1000
   ```

4. **📊 Visualización:**
   ```bash
   python -m HADRA.scripts.dashboard
   ```

5. **🚁 Despliegue de la Misión:**
   ```bash
   python -m HADRA.scripts.mission_deploy --target 10 10 -5 --model-path data/pretrained
   ```

## 🧮 Fundamentación Matemática

📌 **Ecuación de Bellman:**
\[
Q(s, a) = r(s,a) + \gamma \mathbb{E}[Q(s', \pi(s'))]
\]

📌 **Pérdida del Crítico:**
\[
L_{\text{critic}} = \mathbb{E}\left[\left(Q(s,a) - \left(r + \gamma Q_{\text{target}}(s', \pi(s'))\right)\right)^2\right]
\]

📌 **Pérdida del Actor:**
\[
L_{\text{actor}} = -\mathbb{E}\left[Q(s, \pi(s))\right]
\]

📌 **Métricas de Formación:**
- **📏 Cohesión:**
  \[
  C = \frac{1}{N}\sum_{i=1}^{N}\|\mathbf{x}_i - \bar{\mathbf{x}}\|
  \]
- **📉 Varianza de la Formación:**
  \[
  V = \frac{1}{N}\sum_{i=1}^{N}\|\mathbf{x}_i - \bar{\mathbf{x}}\|^2
  \]

## 🔮 Innovación y Futuro

✨ **Meta-Aprendizaje en Tiempo Real:** Ajuste dinámico de hiperparámetros.

🧠 **Integración del LLM para el Agente Madre:** Uso de LLMs (como Qwen/1.5) para mejorar la planificación en tiempo real.

## 🎯 Conclusión

✅ **Planificación Jerárquica** con un agente madre.
✅ **Aprendizaje Actor-Crítico** en microagentes.
✅ **Visualización en Tiempo Real** con dashboards interactivos.

## 🌍 Instrucciones para GitHub

1. **📦 Inicializa el repositorio**
2. **📂 Asegúrate de incluir todos los archivos**
3. **🚫 Agrega un `.gitignore`**
4. **📜 Realiza commits documentados**
5. **🚀 Publica en GitHub y actualiza el README.md**

---

🚀 **HADRA te ofrece una arquitectura de drones autónomos con IA avanzada. ¡Experimenta, optimiza y lidera el futuro de la robótica en enjambres!** 🛸🔥
# 🚀 Arquitectura HADRA (Hybrid Autonomous Drone Reinforcement Agents)

## 📝 Descripción

HADRA es una arquitectura híbrida de agentes de IA diseñada para planificar y ejecutar misiones críticas con un enjambre de drones.

### 🏗️ Componentes:

- **👩‍🚀 Agente Madre de Colmena:** Módulo de planificación global basado en un LLM (por ejemplo, Qwen/1.5).
- **🐝 Microagentes Zánganos (HADRA Units):** Cada dron tiene un microagente con:
  - 🧠 **Núcleo Neuro-Simbólico (NS):** Red neuronal ligera + reglas IF-THEN.
  - 🔧 **Núcleo de Meta-Aprendizaje (ML):** Ajusta hiperparámetros en tiempo real.
  - 👀 **Núcleo de Consciencia Operacional (OA):** Monitoreo del estado y entorno.
  - 🤝 **Núcleo de Cooperación Emergente (EC):** Coordinación y formación en grupo.

## ⚙️ Funcionalidades

- **🎭 Entrenamiento Actor-Crítico:** Microagentes aprenden a alcanzar sub-metas asignadas.
- **📡 Planificación Jerárquica:** Agente madre genera sub-metas.
- **📊 Registro y Visualización:**
  - 📝 Métricas de recompensa y trayectorias en CSV.
  - 📺 Dashboard interactivo (Plotly Dash) para analizar misiones en tiempo real.

## 📂 Estructura del Repositorio
```
HADRA_GPT/
├── HADRA/
│   ├── configs/
│   │   ├── airsim.json
│   │   └── training.yaml
│   ├── core/
│   │   ├── agent.py
│   │   ├── env.py
│   │   ├── hierarchical_agent.py
│   │   ├── micro_agent.py
│   │   └── utils.py
│   └── scripts/
│       ├── airsim_launcher.py
│       ├── deploy.py
│       ├── mission_deploy.py
│       ├── monitor.py
│       ├── train.py
│       └── train_mission.py
├── data/
│   ├── pretrained/  # Modelos guardados
│   └── training_log.csv  # Log de entrenamiento
├── logs/
│   └── telemetry_log.csv  # Telemetría de drones
└── README.md
```

## 🤝 Contribuciones

Este proyecto es desarrollado por **Renzo Valencia Oyarce**, Magíster en Inteligencia Artificial. Si utilizas HADRA en tus investigaciones, proyectos o publicaciones, por favor, da el debido crédito mencionando mi nombre y enlazando este repositorio. Puedes hacerlo de la siguiente manera:

```markdown
📌 Desarrollado por Renzo Valencia Oyarce
🔗 Repositorio: [GitHub.com/TuRepositorio](#)
```

Si deseas contribuir a HADRA, siéntete libre de hacer un fork del repositorio y enviar un pull request con mejoras. Asegúrate de incluir documentación clara sobre los cambios realizados.

Si utilizas esta arquitectura en tu proyecto, por favor, menciona la fuente y da crédito al autor original. Puedes referenciar este repositorio en tus publicaciones o documentación. ¡Toda la comunidad de desarrollo lo agradecerá! 🚀💡

Si deseas contribuir a HADRA, siéntete libre de hacer un fork del repositorio y enviar un pull request con mejoras. Asegúrate de incluir documentación clara sobre los cambios realizados.

Si utilizas esta arquitectura en tu proyecto, por favor, menciona la fuente y da crédito al autor original. Puedes referenciar este repositorio en tus publicaciones o documentación. ¡Toda la comunidad de desarrollo lo agradecerá! 🚀💡

Para cualquier pregunta o sugerencia, no dudes en abrir un issue en GitHub. ¡Felices desarrollos! 🛸

