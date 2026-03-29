# Framework Comparisons

Esta carpeta la reservo para comparar mis implementaciones desde cero con herramientas profesionales.

El objetivo no es "ganar" a `scikit-learn` o `PyTorch`. El objetivo es validar decisiones, medir diferencias de comportamiento y dejar claro que entiendo tanto los fundamentos como las referencias industriales.

## Scripts incluidos

- `sklearn_compare.py`: compara perceptron y red propia con modelos equivalentes de `scikit-learn`.
- `pytorch_compare.py`: compara la red propia con una red pequena construida en `PyTorch`.

## Dependencias opcionales

Estos scripts requieren dependencias externas que ahora mismo no estan instaladas en este entorno:

- `scikit-learn`
- `torch`

Cuando no estan disponibles, los scripts fallan con un mensaje claro y sin afectar al resto del repositorio.
