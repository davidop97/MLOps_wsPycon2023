# 🍷 MLOps Pipeline - Wine Classification

Pipeline completo de MLOps para clasificación de vinos utilizando PyTorch, Weights & Biases (W&B) y GitHub Actions. Este proyecto implementa un flujo de trabajo end-to-end automatizado que incluye carga de datos, preprocesamiento, inicialización del modelo, entrenamiento y evaluación.

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Arquitectura del Pipeline](#arquitectura-del-pipeline)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Ejecución Local](#ejecución-local)
- [CI/CD con GitHub Actions](#cicd-con-github-actions)
- [Resultados y Métricas](#resultados-y-métricas)
- [Capturas de Pantalla](#capturas-de-pantalla)
- [Análisis de Lineage](#análisis-de-lineage)
- [Lecciones Aprendidas](#lecciones-aprendidas)
- [Próximos Pasos](#próximos-pasos)
- [Referencias y Recursos](#referencias-y-recursos)
- [Autor](#autor)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)
- [Agradecimientos](#agradecimientos)
- [Soporte](#soporte)
- [Estado del Proyecto](#estado-del-proyecto)

---

## 🎯 Descripción del Proyecto

Este proyecto implementa un pipeline de Machine Learning Operations (MLOps) para clasificar 3 tipos de vinos basándose en 13 características químicas del dataset Wine de scikit-learn. El enfoque está en la automatización, rastreabilidad y reproducibilidad, utilizando artifacts para gestionar datos y modelos versionados.

### Dataset

- **Nombre**: Wine Recognition Dataset
- **Origen**: scikit-learn
- **Muestras**: 178 (142 entrenamiento, 18 validación, 18 test)
- **Features**: 13 características químicas (alcohol, acidez málica, cenizas, etc.)
- **Clases**: 3 tipos de vinos diferentes

### Objetivo

Construir un clasificador de redes neuronales que prediga el tipo de vino con alta precisión, aplicando prácticas de MLOps para asegurar que el proceso sea auditable y escalable.

---

## 🏗️ Arquitectura del Pipeline

El pipeline se divide en etapas secuenciales, cada una registrada como un artifact en W&B para mantener el lineage:

1. **Carga de Datos** (`data/load.py`)
   - Carga el dataset Wine de scikit-learn.
   - Divide en conjuntos de entrenamiento, validación y test (80%/10%/10%).
   - Guarda los datos como tensors en formato `.pt`.
   - Registra artifact `wine-raw` en W&B.

2. **Preprocesamiento** (`data/preprocess.py`)
   - Descarga el artifact `wine-raw:latest`.
   - Aplica normalización con StandardScaler.
   - Guarda los datos procesados.
   - Registra artifact `wine-preprocess` en W&B.

3. **Inicialización del Modelo** (`build.py`)
   - Define la arquitectura de la red neuronal: `13 → 64 → 32 → 3`.
   - Incluye BatchNorm y Dropout (0.3).
   - Guarda los pesos iniciales.
   - Registra artifact `WineClassifier` en W&B.

4. **Entrenamiento y Evaluación** (`train.py`)
   - Descarga artifacts de datos preprocesados y modelo inicializado.
   - Ejecuta múltiples experimentos con variaciones de hiperparámetros.
   - Registra métricas (loss, accuracy) en W&B.
   - Evalúa en el conjunto de test y identifica ejemplos difíciles.
   - Guarda modelos entrenados como artifacts.

El flujo asegura que cada etapa dependa de la anterior, promoviendo reproducibilidad.

---

## 🛠️ Tecnologías Utilizadas

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| Python | 3.8+ | Lenguaje principal |
| PyTorch | 2.0.1 | Framework de deep learning |
| scikit-learn | 1.3.0 | Dataset y preprocesamiento |
| Weights & Biases | 0.15.4 | Tracking de experimentos y artifacts |
| GitHub Actions | - | CI/CD automatización |
| NumPy | 1.26.4 | Operaciones numéricas |

---

## 📁 Estructura del Proyecto

```
wine-classification-mlops/
├── .github/
│   └── workflows/
│       ├── build_model.yml             # Workflow para inicializar el modelo
│       ├── load_data.yml               # Workflow para cargar datos raw
│       ├── preprocess_data.yml         # Workflow para preprocesar datos
│       ├── testingLoginWandb.yml       # Workflow para probar login en W&B
│       └── train_model.yml             # Workflow para entrenar y evaluar
├── media/                              # Capturas de pantalla
│   ├── build_new_model.png
│   ├── experiments_with_new_model.png
│   ├── GitHubSecret.png
│   ├── load_new_data.png
│   ├── load_raw_artifact.png
│   ├── metrics_best_experiment.png
│   ├── new_classifier_artifact.png
│   ├── preprocess_artifact.png
│   ├── preprocess_new_data.png
│   ├── train_model_successfully.png
│   └── validation_graph_all_experiments.png
├── src/
│   ├── data/
│   │   ├── load.py                     # Carga de datos raw
│   │   └── preprocess.py               # Preprocesamiento de datos
│   ├── __init__.py
│   ├── Classifier.py                   # Definición de la arquitectura del modelo
│   ├── build.py                        # Inicialización del modelo
│   └── train.py                        # Entrenamiento y evaluación
├── README.md                           # Este archivo
└── requirements.txt                    # Dependencias Python
```

---

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- Cuenta en [Weights & Biases](https://wandb.ai/)
- Git

### Pasos de Instalación

1. **Clonar el repositorio**

   ```
   git clone https://github.com/tu-usuario/wine-classification-mlops.git
   cd wine-classification-mlops
   ```

2. **Crear entorno virtual**

   ```
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias**

   ```
   pip install -r requirements.txt
   ```

   **Contenido de `requirements.txt`:**

   ```
   numpy==1.26.4
   matplotlib==3.6.3
   statsmodels==0.13.5
   torch==2.0.1
   torchvision==0.15.2
   wandb==0.15.4
   scikit-learn==1.3.0
   ```

4. **Configurar Weights & Biases**

   ```
   wandb login
   ```

   Ingresa tu API key (obténla desde https://wandb.ai/authorize).

---

## 💻 Ejecución Local

Ejecuta cada script secuencialmente para correr el pipeline localmente.

### Paso 1: Cargar Datos Raw

```
python src/data/load.py
```

**Output esperado aproximado:**
- Información sobre el dataset: tamaños de conjuntos, features y clases.
- Artifact `wine-raw` registrado en W&B.

### Paso 2: Preprocesar Datos

```
python src/data/preprocess.py
```

**Output esperado aproximado:**
- Descarga de artifact `wine-raw:latest`.
- Aplicación de preprocesamiento.
- Artifact `wine-preprocess` registrado en W&B.

### Paso 3: Inicializar Modelo

```
python src/build.py
```

**Output esperado aproximado:**
- Definición y guardado del modelo inicial.
- Artifact `WineClassifier` registrado en W&B.

### Paso 4: Entrenar y Evaluar

```
python src/train.py
```

**Output esperado aproximado:**
- Ejecución de experimentos con métricas registradas.
- Artifacts de modelos entrenados generados en W&B.

---

## 🔄 CI/CD con GitHub Actions

El pipeline se automatiza mediante workflows separados en GitHub Actions, que se ejecutan en pushes o pull requests a `main`, o manualmente. Cada workflow maneja una etapa específica:

- `load_data.yml`: Carga datos raw.
- `preprocess_data.yml`: Preprocesa datos.
- `build_model.yml`: Inicializa el modelo.
- `train_model.yml`: Entrena y evalúa.
- `testingLoginWandb.yml`: Prueba el login en W&B.

### Configuración Común

Cada workflow incluye pasos para checkout, setup de Python, instalación de dependencias y login en W&B usando un secret.

### Configurar Secret en GitHub

1. Ve a Settings > Secrets and variables > Actions en tu repositorio.
2. Agrega `WANDB_API_KEY` con tu API key de W&B.

Los workflows se ejecutan secuencialmente o en paralelo según dependencias configuradas.

---

## 📊 Resultados y Métricas

Se ejecutaron 3 experimentos con configuraciones variadas:

| Experimento | Epochs | Batch Size | Learning Rate | Optimizer | Val Accuracy | Test Accuracy |
|-------------|--------|------------|---------------|-----------|--------------|---------------|
| **001** | 100 | 16 | 0.001 | Adam | 98.59% | **100.00%** |
| **002** | 150 | 16 | 0.0005 | Adam | 98.12% | 100.00% |
| **003** | 200 | 32 | 0.01 | SGD | 97.89% | 97.22% |

El Experimento 001 destacó con 100% de accuracy en test.

---

## 📸 Capturas de Pantalla

Las capturas en `media/` documentan las etapas:

- `build_new_model.png`: Ejecución de workflow para inicializar modelo en GitHub Actions.
- `experiments_with_new_model.png`: Experimentos en dashboard de W&B.
- `GitHubSecret.png`: Configuración de secret `WANDB_API_KEY` en GitHub.
- `load_new_data.png`: Carga de datos nuevos.
- `load_raw_artifact.png`: Artifact de datos raw en W&B.
- `metrics_best_experiment.png`: Métricas del mejor experimento.
- `new_classifier_artifact.png`: Artifact del modelo inicializado.
- `preprocess_artifact.png`: Artifact de preprocesamiento.
- `preprocess_new_data.png`: Preprocesamiento de datos nuevos.
- `train_model_successfully.png`: Entrenamiento exitoso.
- `validation_graph_all_experiments.png`: Gráficas de validación para todos los experimentos.

---

## 🔍 Análisis de Lineage

Weights & Biases rastrea automáticamente la **lineage completa** de cada modelo entrenado:

```
wine-raw:v0
↓ (used by Preprocess Data)
wine-preprocess:v0
↓ (used by Initialize Model)
WineClassifier:v0
↓ (used by Train Model)
trained-wine-model-exp001:v0
```

Esto permite reproducibilidad, auditoría, gobernanza y debugging.

**Cómo visualizarlo en W&B:**
1. Ir a tu proyecto en W&B.
2. Click en pestaña "Artifacts".
3. Seleccionar cualquier artifact (ej: `trained-wine-model-exp001`).
4. Click en "Lineage" para ver el grafo completo.

---

## 🎓 Lecciones Aprendidas

### Mejores Prácticas Implementadas

1. **Versionado de Datos y Modelos**
   - Todos los artifacts versionados automáticamente por W&B.
   - Lineage completo rastreado.

2. **Separación de Concerns**
   - Cada etapa en script separado.

3. **Configuración como Código**
   - Hiperparámetros definidos en código.

4. **CI/CD Automatizado**
   - Workflows en GitHub Actions.

5. **Tracking Exhaustivo**
   - Métricas registradas en tiempo real.

### Desafíos y Soluciones

| Desafío | Solución Implementada |
|---------|----------------------|
| Dataset pequeño | Dropout y BatchNorm para regularización. |
| Overfitting | Learning rate scheduler y validación. |
| Datos tabulares | StandardScaler. |
| Múltiples experimentos | Loops con IDs únicos. |
| Reproducibilidad | Artifacts y random_state fijo. |
| Trazabilidad | Lineage en W&B. |

### Resultados Clave

- Accuracy de 100% en test set.
- Pipeline automatizado.
- Tracking completo.

---

## 🚀 Próximos Pasos

### Mejoras Técnicas

- Implementar hyperparameter tuning con W&B Sweeps.
- Agregar cross-validation.
- Implementar early stopping.
- Exportar a ONNX.
- Agregar tests con pytest.

### MLOps Avanzado

- Model registry.
- Monitoring de drift.
- API REST con FastAPI.
- A/B testing.
- Alertas automáticas.

### Análisis y Visualización

- Dashboard con Streamlit.
- SHAP values.
- Confusion matrix.
- Feature importance.
- Visualización de embeddings.

### Escalabilidad

- Migrar a DVC.
- Distributed training.
- Caching de artifacts.
- Optimizar con Optuna.
- Dockerizar el pipeline.

---

## 📚 Referencias y Recursos

### Documentación Oficial

- [Weights & Biases - Artifacts](https://docs.wandb.ai/guides/artifacts)
- [PyTorch - Data Loading Tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [scikit-learn - Wine Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset)
- [GitHub Actions - Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)

### Mejores Prácticas MLOps

- [MLOps Best Practices - Neptune.ai](https://neptune.ai/blog/mlops-best-practices)
- [CI/CD for Machine Learning - W&B Course](https://wandb.ai/site/courses/cicd/)
- [Designing MLOps Pipelines](https://domino.ai/blog/designing-a-best-in-class-mlops-pipeline)

### Artículos y Tutoriales

- [Structuring ML Projects with MLOps](https://towardsdatascience.com/structuring-your-machine-learning-project-with-mlops-in-mind-41a8d65987c9/)
- [MLOps: Continuous Delivery - Google Cloud](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

---

## 👥 Autor

**David Oliva**
- GitHub: [@davidop97](https://github.com/davidop97)
- LinkedIn: [David Oliva Patiño](www.linkedin.com/in/david-oliva-patino)

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Para contribuir:

1. Fork el repositorio.
2. Crea una rama (`git checkout -b feature/AmazingFeature`).
3. Commit cambios (`git commit -m 'Add some AmazingFeature'`).
4. Push (`git push origin feature/AmazingFeature`).
5. Abre un Pull Request.

Áreas para contribuir:
- Nuevos datasets.
- Nuevas arquitecturas.
- Mejorar visualizaciones.
- Optimizar hiperparámetros.
- Agregar tests.

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver archivo `LICENSE` para más detalles.

MIT License

Copyright (c) 2025 David Oliva

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 🙏 Agradecimientos

- **scikit-learn** por el dataset Wine.
- **Weights & Biases** por la plataforma de tracking.
- **PyTorch Team** por el framework.
- **GitHub** por CI/CD.
- Inspirado en tutoriales de MLOps.

---

## 📞 Soporte

Si tienes preguntas:
1. Abre un Issue en GitHub.
2. Contacta al autor.

---

## 🎯 Estado del Proyecto

**Status**: ✅ Producción

- [x] Pipeline implementado.
- [x] CI/CD configurado.
- [x] Documentación completa.
- [x] Capturas de pantalla.
- [x] Modelo con 100% accuracy.
- [x] Artifacts en W&B.
- [ ] Deployment en producción.
- [ ] API REST.

---

## 📈 Métricas del Proyecto

- **Líneas de código**: ~500
- **Scripts**: 4 (load, preprocess, build, train)
- **Experimentos**: 3
- **Mejor accuracy**: 100%
- **Artifacts**: 7
- **Duración del pipeline**: ~5 minutos

---

**⭐ Si este proyecto te fue útil, dale una estrella en GitHub!**

---

**Última actualización**: Noviembre 2025
````