# AI Lab

Este repositorio es mi laboratorio personal de inteligencia artificial y aprendizaje automatico desde cero. Lo estoy construyendo con una intencion concreta: entender de verdad que ocurre dentro de un perceptron, una neurona, una capa densa y una red neuronal, sin esconder la logica detras de frameworks de alto nivel.

No quiero que este proyecto sea una coleccion de scripts sueltos. Quiero que sea un repositorio que demuestre criterio tecnico, progresion real y capacidad para disenar software entendible, escalable y honesto.

## Snapshot

Este proyecto demuestra que puedo traducir fundamentos de machine learning a software mantenible:

- redes neuronales densas implementadas desde cero con NumPy
- backpropagation vectorizado sobre multiples capas
- optimizadores con estado: SGD, Momentum, RMSprop, Adadelta, Adam, AdamW y Adamax
- regularizacion con dropout y L2
- validacion durante entrenamiento y early stopping
- persistencia de modelos con guardado/carga de arquitectura, pesos y clases
- callbacks reutilizables para logging, checkpoints, learning rate scheduling y early stopping
- metricas de evaluacion: accuracy, confusion matrix, precision, recall, F1, R2 y MAE
- capas adicionales: flatten, batch normalization, convolucion 2D y max pooling 2D
- gradient checking reutilizable para validar derivadas
- benchmarks reproducibles y ejemplos con datasets clasicos
- suite automatizada con tests
- empaquetado instalable con `pyproject.toml`
- CI preparado con GitHub Actions
- demos y graficos reproducibles en `artifacts/plots`

La intencion no es sustituir PyTorch. La intencion es demostrar comprension profunda, criterio de diseno y disciplina de ingenieria.

## Vista rapida

- Estado: funcional y testeado
- Stack: Python, NumPy, matplotlib, unittest, setuptools
- Enfoque: machine learning y redes neuronales desde cero
- Objetivo: aprender fundamentos y construir una base de software mantenible
- Cobertura actual: perceptron, neurona diferenciable, capas densas, dropout, red neuronal, optimizadores, validacion, early stopping, persistencia, visualizacion, ejemplos y comparaciones externas

## Capacidades actuales

| Area | Estado actual |
|---|---|
| Perceptron binario | Implementado y probado |
| Neurona diferenciable | Implementada y probada |
| Capas densas | Implementadas con forward y backward |
| Dropout | Implementado como capa reutilizable |
| Batch normalization | Implementada como capa reutilizable |
| Capas CNN iniciales | `Conv2DLayer` y `MaxPool2DLayer` con forward NumPy |
| Red neuronal multicapa | Implementada con mini-batches, backpropagation y optimizadores configurables |
| Activaciones | `identity`, `sigmoid`, `tanh`, `relu`, `softmax` |
| Optimizadores | `sgd`, `momentum`, `rmsprop`, `adadelta`, `adam`, `adamw`, `adamax` |
| Validacion | `validation_split`, `validation_data`, `val_loss`, `val_metric` |
| Early stopping | Parada por falta de mejora con `patience`, `min_delta` y restauracion opcional |
| Callbacks | `EarlyStopping`, `ModelCheckpoint`, `LearningRateScheduler`, `HistoryLogger` |
| Persistencia | `save()` y `NeuralNetwork.load()` para modelos entrenados |
| Metricas | Accuracy, confusion matrix, precision, recall, F1, R2 y MAE |
| Gradient checking | Utilidad reusable para comprobar gradientes de capas densas |
| Problemas soportados | Regresion, clasificacion binaria y multiclase |
| Visualizacion | Historicos de entrenamiento, comparacion 2D y comparacion entre optimizadores |
| Examples | Demo real sobre MNIST con red densa |
| Benchmarks | Tabla reproducible con tareas sinteticas y comparacion opcional sklearn |
| Tests | Suite automatizada activa |
| CI | Workflow de GitHub Actions para tests y lint |
| Comparaciones externas | Scripts base para `scikit-learn` y `PyTorch` |

## Que estoy construyendo

Estoy desarrollando una base de modelos y componentes de redes neuronales en Python y NumPy, priorizando:

- claridad matematica
- codigo legible
- separacion de responsabilidades
- vectorizacion cuando aporta valor real
- posibilidad de escalar a arquitecturas y experimentos mas complejos

El proyecto no pretende competir con PyTorch o TensorFlow. Pretende demostrar que entiendo el problema, se modelarlo bien y puedo traducir teoria en codigo serio.

## Estado actual del proyecto

Ahora mismo el repositorio ya tiene una base funcional y testeada:

- Un `Perceptron` binario clasico con `fit`, `partial_fit`, `predict`, `score`, `decision_function` y `margin`.
- Un nucleo modular en `from-scratch/neural_core` para modelos diferenciables.
- Activaciones desacopladas con soporte para `identity`, `sigmoid`, `tanh`, `relu` y `softmax`.
- Una clase `DenseLayer` para capas totalmente conectadas con forward y backward vectorizados.
- Una clase `DropoutLayer` para regularizacion en capas ocultas.
- Una clase `NeuralNetwork` con mini-batches, backpropagation, perdidas para regresion, binaria y multiclase, optimizadores configurables, validacion, early stopping, persistencia, y metodos `forward`, `predict`, `predict_proba` y `score`.
- Una clase `Neuron` como caso particular de una sola neurona diferenciable, reutilizando la infraestructura general.
- Un modulo de optimizadores con soporte para `SGD`, `Momentum`, `RMSprop`, `Adadelta`, `Adam`, `AdamW` y `Adamax`.
- Un script de demo con comparacion `Perceptron` vs `Neuron`, comparativa de optimizadores, una red multiclase con dropout y una red para XOR.
- Una carpeta `examples/` para demos mas orientadas al usuario, incluyendo MNIST.
- Utilidades de visualizacion para historicos de entrenamiento, comparaciones en 2D y comparativas entre optimizadores.
- Una suite de tests automatizados para activaciones, capas, dropout, perceptron, neurona, red y visualizacion.
- Una carpeta `comparisons/` para contrastar el comportamiento del laboratorio con `scikit-learn` y `PyTorch`.

En el estado actual, la suite automatica del proyecto pasa completa con:

```bash
python -m unittest discover -s tests -v
```

Resultado actual:

```text
Ran 42+ tests
OK
```

## Arranque rapido

Si alguien entra al repositorio por primera vez, este es el recorrido mas corto para entenderlo:

```bash
python -m pip install -e .
python from-scratch/NeuralDemo.py
python examples/mnist_demo.py --train-size 12000 --test-size 2000 --epochs 12
python examples/classic_datasets_demo.py
python benchmarks/run_benchmarks.py
python -m unittest discover -s tests -v
```

Con eso ya se puede ver:

- una comparacion entre perceptron y neurona
- una comparativa de optimizadores sobre un problema de regresion
- una red multiclase con `DropoutLayer`
- una red densa resolviendo XOR
- una demo mas realista sobre MNIST
- demos sobre Iris, Breast Cancer y Diabetes si `scikit-learn` esta instalado
- una tabla de benchmarks reproducible
- generacion automatica de graficos
- validacion automatizada del nucleo

## Instalacion como paquete

El repositorio incluye `pyproject.toml`, asi que tambien puede instalarse en modo editable:

```bash
python -m pip install -e .
```

Para desarrollo local con lint:

```bash
python -m pip install -e ".[dev]"
python -m ruff check from-scratch tests examples comparisons benchmarks
```

Tambien hay un comando CLI para benchmarks:

```bash
ai-lab-benchmark
```

Si el directorio de scripts de Python no esta en `PATH`, usa:

```bash
python -m benchmarks.run_benchmarks
```

## Documentacion tecnica

La carpeta `docs/` contiene notas cortas orientadas a explicar decisiones:

- `docs/architecture.md`
- `docs/backpropagation.md`
- `docs/optimizers.md`
- `docs/persistence.md`
- `docs/benchmarks.md`
- `docs/frontend_plan.md`

Tambien hay un notebook de entrada en `notebooks/01_training_a_network_from_scratch.ipynb`.

## Requisitos

Para trabajar con el repositorio en su estado actual basta con tener:

- Python 3.11 o superior
- `numpy`
- `matplotlib`

Dependencias opcionales para comparaciones externas:

- `scikit-learn`
- `torch`

Dependencias opcionales para la demo de MNIST:

- `tensorflow`
- `keras`
- `scikit-learn`

## Estructura actual

```text
AI - lab/
|-- README.md
|-- .gitignore
|-- artifacts/
|   `-- plots/
|       |-- mnist_sample_predictions.png
|       |-- mnist_training_history.png
|       |-- multiclass_dropout_history.png
|       |-- neuron_training_history.png
|       |-- optimizer_regression_comparison.png
|       |-- perceptron_training_history.png
|       |-- perceptron_vs_neuron.png
|       `-- xor_network_history.png
|-- comparisons/
|   |-- README.md
|   |-- common.py
|   |-- pytorch_compare.py
|   `-- sklearn_compare.py
|-- examples/
|   |-- README.md
|   `-- mnist_demo.py
|-- from-scratch/
|   |-- NeuralDemo.py
|   |-- perceptron.py
|   |-- neural_core/
|   |   |-- __init__.py
|   |   |-- activations.py
|   |   |-- common.py
|   |   |-- layers.py
|   |   |-- network.py
|   |   |-- neuron.py
|   |   `-- optimizers.py
|   `-- visualization/
|       |-- __init__.py
|       `-- training_plots.py
`-- tests/
    |-- _bootstrap.py
    |-- test_activations.py
    |-- test_layers.py
    |-- test_network.py
    |-- test_neuron.py
    |-- test_perceptron.py
    `-- test_visualization.py
```

## Que hace cada parte

### `from-scratch/perceptron.py`

Aqui mantengo una implementacion de perceptron binario clasico. Es la pieza fundacional para explicar frontera lineal, margen y actualizacion basada en errores sin mezclar todavia el enfoque del gradiente.

Tambien sirve como punto de comparacion frente a la neurona diferenciable y frente a redes mas profundas.

### `from-scratch/neural_core/activations.py`

Aqui centralizo las funciones de activacion y sus derivadas. Esta separacion evita mezclar la matematica de activacion con la logica de entrenamiento o con el almacenamiento de parametros.

### `from-scratch/neural_core/layers.py`

Aqui viven `DenseLayer` y `DropoutLayer`.

`DenseLayer` representa una capa totalmente conectada y almacena pesos, sesgos y caches del forward para poder hacer backpropagation de forma limpia.

`DropoutLayer` introduce regularizacion como una capa no parametrica reutilizable, sin forzar hacks dentro de `NeuralNetwork`.

La implementacion esta vectorizada, de modo que el salto desde una intuicion de neuronas individuales a una arquitectura matricial real sea natural.

### `from-scratch/neural_core/optimizers.py`

Aqui concentro la logica de actualizacion de parametros. Esta separacion evita mezclar reglas de optimizacion con la definicion de capas.

Ahora mismo el proyecto soporta:

- `SGD`
- `Momentum`
- `RMSprop`
- `Adadelta`
- `Adam`
- `AdamW`
- `Adamax`

### `from-scratch/neural_core/network.py`

Esta es la pieza central del proyecto actual.

La clase `NeuralNetwork` ya resuelve:

- entrenamiento por epocas
- mini-batches
- mezcla aleatoria de datos
- inicializacion reproducible con `random_state`
- soporte para regresion, clasificacion binaria y multiclase
- validacion y preparacion de `X` e `y`
- forward, prediccion y scoring
- backpropagation sobre varias capas
- capas parametricas y no parametricas
- optimizadores configurables sin romper la API principal

La API ya esta preparada para crecer hacia optimizadores, regularizacion adicional, persistencia y mejores experimentos sin tener que rehacer el nucleo.

### `from-scratch/neural_core/neuron.py`

Aqui encapsulo una sola neurona diferenciable como caso particular de la red general. Esto evita duplicar logica de entrenamiento y mantiene el proyecto mas coherente.

### `from-scratch/NeuralDemo.py`

Este archivo es el punto de entrada principal para demos manuales del proyecto. Actualmente incluye:

- una comparacion entre `Perceptron` y `Neuron` sobre un problema binario sencillo
- una comparativa entre varios optimizadores sobre una regresion lineal
- una red multiclase con `DropoutLayer`
- una red neuronal densa para resolver XOR
- guardado automatico de graficos en `artifacts/plots`

### `examples/mnist_demo.py`

Este script usa el nucleo del proyecto para entrenar una red densa multiclase sobre MNIST.

Incluye:

- carga de MNIST desde varias fuentes opcionales
- preprocesado simple de imagenes `28x28` a vectores de `784`
- entrenamiento con `NeuralNetwork`, `DenseLayer`, `DropoutLayer` y `Adam`
- grafico de entrenamiento y una vista previa de predicciones

La idea aqui no es competir con una CNN, sino demostrar que el nucleo actual ya puede enfrentarse a un dataset real y conocido.

### `from-scratch/visualization/training_plots.py`

Aqui concentro la generacion de graficos del proyecto con `matplotlib`.

Ahora mismo las utilidades permiten:

- pintar historicos de entrenamiento a partir de distintos tipos de snapshot
- comparar perceptron y neurona sobre un problema binario en 2D
- comparar curvas de entrenamiento entre varios optimizadores
- guardar figuras automaticamente en `artifacts/plots`

### `tests/`

La carpeta de tests existe para que el laboratorio no dependa solo de demos manuales. La suite actual valida:

- funciones de activacion y sus derivadas
- construccion, forward y backward de capas densas
- contrato de `DropoutLayer`
- una comprobacion numerica del backward en `DenseLayer`
- perceptron clasico
- neurona diferenciable
- red neuronal en problemas de regresion, binarios y multiclase
- soporte real de varios optimizadores
- generacion de figuras

### `comparisons/`

Esta carpeta separa las comparaciones contra frameworks profesionales del nucleo hecho desde cero.

He dejado scripts base para:

- `scikit-learn`
- `PyTorch`

Si esas dependencias no estan instaladas, los scripts terminan con un mensaje claro y no afectan al resto del repositorio.

## Que demuestra este proyecto a nivel tecnico

Las decisiones que ya refleja el codigo son:

- separacion entre implementaciones fundacionales y nucleo reutilizable
- modulos pequeños con responsabilidad clara
- traduccion de conceptos matematicos a una API razonable
- uso de NumPy de forma vectorizada cuando tiene sentido
- una base preparada para crecer sin convertirse en una acumulacion de scripts

Tambien hay una decision deliberada en lo que todavia no estoy haciendo: no he corrido a envolver esto en una interfaz bonita ni a vender humo con metricas artificiales. Primero quiero una base solida.

## Como ejecutar el proyecto

Desde la raiz del repositorio:

```bash
python from-scratch/perceptron.py
python from-scratch/NeuralDemo.py
python examples/mnist_demo.py --train-size 12000 --test-size 2000 --epochs 12
python -m unittest discover -s tests -v
python comparisons/sklearn_compare.py
python comparisons/pytorch_compare.py
```

Los scripts actuales permiten verificar:

- clasificacion binaria simple con perceptron
- comparacion entre perceptron y neurona diferenciable
- comparacion entre optimizadores sobre regresion
- uso de `DropoutLayer` en una red multiclase
- resolucion de XOR mediante una red densa
- una demo realista de clasificacion multiclase sobre MNIST
- generacion automatica de graficos en `artifacts/plots`
- validacion automatizada del nucleo

## Artefactos generados

El proyecto ya genera figuras reutilizables en `artifacts/plots`, utiles tanto para depuracion como para documentacion visual del repositorio:

- `perceptron_vs_neuron.png`
- `perceptron_training_history.png`
- `neuron_training_history.png`
- `optimizer_regression_comparison.png`
- `multiclass_dropout_history.png`
- `xor_network_history.png`
- `mnist_training_history.png`
- `mnist_sample_predictions.png`

Esto ayuda a que el repositorio no solo diga que el entrenamiento funciona, sino que tambien lo muestre.

## Filosofia del repositorio

Estoy construyendo este laboratorio con una filosofia simple:

- primero entender
- despues abstraer
- despues optimizar
- despues escalar

No quiero saltarme capas de comprension. Cuando uso PyTorch, quiero hacerlo con criterio, sabiendo exactamente que resuelve el framework y que compromisos estoy aceptando.

## Limitaciones actuales

No intento esconder lo que todavia falta. Ahora mismo faltan, o estan verdes, varias piezas importantes:

- callbacks de entrenamiento mas completos mas alla de early stopping
- regularizacion mas completa
- persistencia mas avanzada incluyendo estado del optimizador y metadatos de experimentos
- una separacion aun mas formal entre libreria, demos y experimentos
- benchmarks y comparativas mas exigentes
- datasets algo mas representativos
- documentacion de API y ejemplos mas amplia
- visualizaciones mas ricas, incluyendo mas casos multiclase y comparativas

## Roadmap visible

- [x] Implementar un perceptron binario desde cero
- [x] Implementar una neurona diferenciable reutilizando infraestructura comun
- [x] Implementar capas densas con forward y backward
- [x] Implementar una red neuronal multicapa con mini-batches
- [x] Anadir optimizadores configurables al nucleo
- [x] Introducir `DropoutLayer` como capa reutilizable
- [x] Anadir visualizacion de entrenamiento
- [x] Anadir tests automatizados del nucleo
- [x] Preparar comparaciones base con frameworks externos
- [x] Incorporar persistencia de modelos
- [x] Incorporar validacion durante entrenamiento
- [x] Incorporar early stopping
- [x] Anadir empaquetado instalable con `pyproject.toml`
- [x] Preparar CI con GitHub Actions
- [x] Introducir nuevas capas o abstracciones de capa adicionales
- [x] Anadir demos oficiales adicionales de regresion y multiclase
- [x] Mejorar comparativas y benchmarks
- [x] Anadir callbacks reutilizables
- [x] Anadir metricas de evaluacion
- [x] Anadir gradient checking
- [x] Anadir documentacion tecnica corta
- [x] Anadir notebook de entrada
- [ ] Refinar la separacion entre libreria, demos y experimentos

## Hacia donde puede avanzar de forma logica

La evolucion natural del proyecto no es anadir clases sin criterio, sino ampliar el nucleo actual en el orden correcto:

### 1. Reforzar `DenseLayer` y `NeuralNetwork`

Lo mas logico es seguir ampliando las clases que ya sostienen el proyecto:

- regularizacion adicional
- callbacks adicionales
- persistencia avanzada del modelo

### 2. Formalizar mejor la API

Antes de multiplicar tipos de modelo, conviene consolidar mejor la API publica y dejar mas claro que parte del codigo es nucleo reutilizable y que parte son demos.

### 3. Introducir nuevas capas

Cuando el nucleo este mas firme, el siguiente salto razonable es anadir nuevas capas o variantes, por ejemplo:

- una interfaz base mas explicita para capas
- capas adicionales de regularizacion o normalizacion
- bloques mas reutilizables para arquitecturas futuras

### 4. Subir el nivel experimental

Despues tiene sentido ampliar demos y comparativas:

- regresion mas visible
- multiclase con datasets sinteticos
- comparativas mas limpias frente a `scikit-learn` y `PyTorch`
- exploracion de activaciones, tamanos de capa y batch size

### 5. Endurecer tests y medicion

Una vez que el nucleo crezca, toca validar mejor:

- shapes y errores esperados
- estabilidad numerica basica
- comportamiento de gradientes
- coste por epoca y efecto del batch size

## Proximos pasos inmediatos razonables

Si quiero subir el nivel del repositorio sin romper foco, los siguientes pasos mas coherentes son:

- ampliar tests numericos y de contrato
- documentar mejor la API publica
- anadir mas demos oficiales y datasets pequeños pero mas variados
- enriquecer la carpeta `comparisons/`
- limpiar estructura y nombres si el proyecto sigue creciendo

## Por que este repositorio puede ser util

Este proyecto no esta pensado solo como ejercicio academico. Tambien sirve como:

- laboratorio para entender backpropagation y entrenamiento paso a paso
- base para experimentar con nuevas capas y optimizadores
- portfolio tecnico con decisiones de arquitectura visibles
- punto de comparacion entre fundamentos implementados a mano y tooling industrial

## Conclusion

Estoy construyendo este proyecto como laboratorio, como portfolio y como prueba de disciplina tecnica. La base actual ya funciona, ya esta testeada y ya tiene una arquitectura que merece la pena seguir ampliando con criterio.
