# Persistence

Fitted models can be saved and loaded:

```python
model.save("artifacts/models/xor_model.npz")
loaded = NeuralNetwork.load("artifacts/models/xor_model.npz")
```

The file stores:

- model configuration
- layer configuration
- dense weights and biases
- batch normalization state
- class labels for classifiers

Current limitation: optimizer runtime state is not persisted yet. Loaded models are
intended for inference or continued training with fresh optimizer state.
