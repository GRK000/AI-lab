# AI Lab

Este repositorio es mi laboratorio personal de inteligencia artificial y aprendizaje automatico desde cero. Lo estoy construyendo con una intencion muy concreta: entender de verdad que ocurre dentro de un perceptron, una neurona, una capa densa y una red neuronal, sin esconder la logica detras de frameworks de alto nivel.

No quiero que este proyecto sea una coleccion de scripts sueltos. Quiero que sea un repositorio que demuestre criterio tecnico, progresion real y capacidad para disenar software entendible, escalable y honesto. Si alguien llega aqui desde GitHub, mi objetivo es que vea dos cosas muy rapido:

1. Que se implementar fundamentos de machine learning sin depender de magia externa.
2. Que se organizar ese trabajo como software mantenible, legible y preparado para crecer.

## Que estoy construyendo

Estoy desarrollando una base de modelos y componentes de redes neuronales desde cero, en Python y con NumPy, priorizando:

- claridad matematica
- codigo legible
- separacion de responsabilidades
- posibilidad real de escalar a arquitecturas mas complejas
- compatibilidad con futuras mejoras como tests, benchmarks y documentacion tecnica mas profunda

El proyecto no pretende competir con PyTorch o TensorFlow. Pretende demostrar que entiendo el problema, se modelarlo bien y puedo traducir teoria en codigo serio.

## Estado actual del proyecto

Hasta ahora he implementado:

- Un `Perceptron` binario clasico, entrenado con la regla del perceptron.
- Una base modular para redes densas en `from-scratch/neural_core`.
- Un sistema de activaciones desacoplado del resto de la logica.
- Una clase `DenseLayer` para representar capas totalmente conectadas.
- Una clase `NeuralNetwork` con entrenamiento por mini-batches y backpropagation.
- Una clase `Neuron` como wrapper de una sola neurona diferenciable para problemas simples.
- Demos sencillas para clasificacion binaria y resolucion de XOR.
- Una carpeta `tests/` con validacion automatizada de activaciones, capas, perceptron, neurona y red.
- Utilidades de visualizacion con `matplotlib` para historicos de entrenamiento y comparacion entre modelos.
- Una carpeta `comparisons/` para contrastar el comportamiento del laboratorio con frameworks profesionales.

## Estructura actual

```text
AI - lab/
|-- README.md
|-- artifacts/
|   `-- plots/
|-- comparisons/
|   |-- README.md
|   |-- common.py
|   |-- sklearn_compare.py
|   `-- pytorch_compare.py
|-- tests/
|   |-- _bootstrap.py
|   |-- test_activations.py
|   |-- test_layers.py
|   |-- test_network.py
|   |-- test_neuron.py
|   |-- test_perceptron.py
|   `-- test_visualization.py
|-- from-scratch/
|   |-- perceptron.py
|   |-- perceptron-binario.py
|   |-- NeuralDemo.py
|   |-- neural_core/
|   |   |-- __init__.py
|   |   |-- common.py
|   |   |-- activations.py
|   |   |-- layers.py
|   |   |-- network.py
|   |   `-- neuron.py
|   `-- visualization/
|       |-- __init__.py
|       `-- training_plots.py
```

## Que hace cada parte

### `from-scratch/perceptron.py`

Aqui mantengo una implementacion de perceptron binario clasico. Me interesa porque es una pieza fundacional: separa con claridad la intuicion de frontera lineal, el concepto de margen y la actualizacion basada en errores.

No es solo un ejercicio academico. Tambien sirve como punto de comparacion frente a la neurona entrenada por gradiente y frente a redes mas profundas.

### `from-scratch/neural_core/activations.py`

Aqui centralizo las funciones de activacion y sus derivadas. Tome esta decision porque no quiero mezclar la matematica de activacion con la logica de entrenamiento o con el almacenamiento de parametros.

Ahora mismo soporta:

- `identity`
- `sigmoid`
- `tanh`
- `relu`
- `softmax`

Esto me permite cambiar el comportamiento de una capa sin reescribir la arquitectura completa.

### `from-scratch/neural_core/layers.py`

Aqui esta `DenseLayer`, que representa una capa totalmente conectada. Internamente almacena pesos, sesgos y caches del forward para poder hacer backpropagation de forma limpia.

La decision importante aqui fue vectorizar la implementacion. Aunque conceptualmente hablo de neuronas, el calculo se hace con matrices para mantener eficiencia y para que el salto a redes mayores sea natural.

### `from-scratch/neural_core/network.py`

Esta es la pieza central del proyecto actual.

La clase `NeuralNetwork` ya resuelve varios problemas relevantes:

- entrenamiento por epocas
- mini-batches
- mezcla aleatoria de datos
- inicializacion reproducible con `random_state`
- perdidas para regresion, binaria y multiclase
- `forward`, `predict`, `predict_proba` y `score`
- preparacion y validacion de targets
- backpropagation sobre varias capas

Mi intencion aqui no era solo "hacer que funcione", sino disenar una API que no me obligue a tirar codigo cuando quiera anadir optimizadores, regularizacion adicional, metricas o persistencia.

### `from-scratch/neural_core/neuron.py`

Aqui encapsulo una sola neurona diferenciable como un caso particular de la red general. Lo hice asi porque una neurona aislada es util para explicar ideas, pero no queria duplicar logica de entrenamiento.

En otras palabras: la neurona simple reutiliza la infraestructura correcta, en lugar de convertirse en una excepcion mal mantenida.

### `from-scratch/NeuralDemo.py` y `from-scratch/perceptron-binario.py`

Estos archivos actuan como puntos de entrada y ejemplos de uso. Para mi son importantes porque un repositorio orientado a portfolio no debe obligar a quien lo visita a recorrer toda la base de codigo para entender si algo funciona o no.

### `from-scratch/visualization/training_plots.py`

Aqui concentro la generacion de graficos del proyecto. Uso `matplotlib` porque mi objetivo no es reinventar una libreria de visualizacion, sino analizar entrenamiento y producir artefactos utiles para documentar el comportamiento de los modelos.

Ahora mismo estas utilidades sirven para:

- pintar historicos de entrenamiento
- comparar perceptron y neurona sobre un problema binario en 2D
- guardar las figuras automaticamente en `artifacts/plots`

### `tests/`

La carpeta de tests existe para que el laboratorio deje de depender solo de demos manuales. Quiero poder cambiar activaciones, inicializaciones o detalles del entrenamiento con la tranquilidad de que una suite automatizada detectara regresiones.

La suite valida:

- funciones de activacion y sus derivadas
- construccion y backward de capas densas
- perceptron clasico
- neurona diferenciable
- red neuronal en problemas de regresion, binarios y multiclase
- generacion de figuras

### `comparisons/`

Esta carpeta separa la comparacion contra frameworks profesionales del nucleo hecho desde cero. Para mi es importante porque demuestra dos cosas: que entiendo los fundamentos y que se contrastarlos contra herramientas industriales.

He dejado scripts base para:

- `scikit-learn`
- `PyTorch`

Si esas dependencias no estan instaladas, los scripts terminan con un mensaje claro y no afectan al resto del repositorio.

## Que demuestra este proyecto a nivel tecnico

Este repositorio esta pensado para enseñar más que resultados. Quiero que quien lo vea entienda como pienso al programar.

Las decisiones que ya refleja el codigo son:

- Se separar una implementacion experimental de un nucleo reutilizable.
- Se disenar modulos pequenos con una responsabilidad clara.
- Se traducir conceptos matematicos a una API de software razonable.
- Se priorizar codigo vectorizado cuando tiene sentido.
- Se dejar la base preparada para escalar, en lugar de encadenar scripts irreutilizables.

Tambien hay una decision deliberada en lo que no estoy haciendo todavia: no he corrido a envolver esto en una interfaz bonita ni a vender humo con metricas artificiales. Primero quiero una base solida.

## Como ejecutar el proyecto

Desde la raiz del repositorio:

```bash
python from-scratch/perceptron.py
python from-scratch/perceptron-binario.py
python from-scratch/NeuralDemo.py
python -m unittest discover -s tests -v
python comparisons/sklearn_compare.py
python comparisons/pytorch_compare.py
```

Los scripts actuales muestran ejemplos pequenos para verificar:

- clasificacion binaria simple
- uso de una neurona diferenciable
- resolucion de XOR mediante una red densa
- generacion automatica de graficos en `artifacts/plots`
- validacion automatizada del nucleo

## Filosofia del repositorio

Estoy construyendo este laboratorio con una filosofia simple:

- primero entender
- despues abstraer
- despues optimizar
- despues escalar

No quiero saltarme capas de comprension. Cuando uso PyTorch, quiero hacerlo con criterio, sabiendo exactamente que resuelve el framework y que compromisos estoy aceptando.

## Limitaciones actuales

No intento esconder lo que todavia falta. Un proyecto serio mejora mucho cuando deja claras sus fronteras actuales.

Ahora mismo faltan, o estan verdes, varias piezas importantes:

- benchmarks mas amplios y comparativas mas exigentes
- tipado aun mas estricto en algunos puntos
- documentacion de ejemplos mas amplia
- separacion formal entre codigo de libreria y scripts de demo
- optimizadores adicionales como SGD con momentum o Adam
- regularizacion mas completa
- guardado y carga de modelos
- visualizaciones mas ricas, incluyendo fronteras multiclase y curvas comparativas adicionales
- datasets algo mas realistas

## Que hare para mejorar el proyecto

Si quiero que este repositorio suba de nivel y no se quede en un "experimento simpatico", estos son los siguientes pasos naturales:

### 1. Profundizar la estrategia de tests

Quiero validar no solo que los ejemplos funcionan, sino que:

- las dimensiones son correctas
- las perdidas bajan donde deben
- los gradientes responden como espero
- la API falla bien cuando recibe entradas invalidas
- las comparativas externas mantienen una referencia razonable

### 2. Separar demos, libreria y experimentos

Mi siguiente refactor razonable es dejar tres zonas claras:

- `core/` para la logica reutilizable
- `examples/` para casos ejecutables
- `experiments/` para pruebas mas libres

Eso haria que el repositorio respire mejor y sea mas facil de evaluar.

### 3. Ampliar experimentos y comparativas

No basta con decir "funciona". Quiero incluir comparativas pequenas pero honestas:

- perceptron clasico vs neurona sigmoide
- una capa vs dos capas
- distintas activaciones
- distintos tamanos de capa oculta
- comparacion contra `scikit-learn` y `PyTorch` con datasets mas representativos

Eso convertiria el repo en una herramienta de aprendizaje y tambien en una demostracion de criterio experimental.

### 4. Mejorar la presentacion para GitHub

Para que el proyecto gane fuerza como portfolio, quiero anadir:

- imagenes o diagramas sencillos de arquitectura
- tablas cortas con capacidades actuales
- un changelog o roadmap visible
- ejemplos de salida esperada
- una seccion de "lecciones aprendidas"

Un buen proyecto no solo se programa bien; tambien se comunica bien.

### 5. Incorporar datasets mas representativos

XOR esta bien como prueba conceptual, pero no basta. El proyecto sube mucho cuando demuestra comportamiento en datos algo mas serios, aunque sigan siendo pequenos:

- clasificacion lineal y no lineal sintetica
- regresion simple
- multiclase en datos generados
- comparacion basica con una implementacion de referencia

### 6. Medir y perfilar

Si de verdad quiero que esto transmita nivel, necesito enseñar que tambien se mirar rendimiento. NumPy ya da una buena base, pero quiero medir:

- coste por epoca
- impacto del tamano de batch
- diferencias entre configuraciones
- cuellos de botella del codigo

## Por que este proyecto merece la pena

Este repositorio me sirve para aprender, pero no solo para aprender. Tambien me sirve para demostrar algo que valoro mucho en ingenieria:

soy capaz de empezar por fundamentos, detectar cuando un script ya no escala, refactorizarlo en piezas limpias y dejar preparado el terreno para lo siguiente.

Eso, para mi, es mucho mas interesante que tener una demo vistosa montada deprisa.


## Proximos pasos inmediatos

Mi siguiente iteracion probable sera una combinacion de estas tareas:

- ampliar tests a casos numericos mas exigentes
- enriquecer la carpeta de comparaciones externas
- documentar mejor la API publica
- incorporar al menos un caso de regresion y uno multiclase como demos oficiales
- limpiar nombres de archivos para que la estructura sea aun mas consistente
- generar imagenes reutilizables para el README y el perfil de GitHub

## Conclusión

Estoy construyendo este proyecto como laboratorio, como portfolio y como prueba de disciplina tecnica. Prefiero que cada parte este entendida y justificada antes de seguir anadiendo complejidad.
