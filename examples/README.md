# Examples

Esta carpeta contiene ejemplos ejecutables orientados al usuario.

## Ejemplos actuales

- `mnist_demo.py`: entrena una red densa sobre MNIST usando el nucleo del proyecto.
- `classic_datasets_demo.py`: entrena modelos sobre Iris, Breast Cancer y Diabetes usando `scikit-learn` solo para cargar y partir datasets.

## Notas

- `mnist_demo.py` intenta cargar MNIST desde varias fuentes opcionales.
- Si no encuentra una fuente disponible, termina con un mensaje claro y explica que dependencia instalar o que comando probar.
- `classic_datasets_demo.py` requiere `scikit-learn`:

```bash
python -m pip install -e ".[examples]"
python examples/classic_datasets_demo.py
```
