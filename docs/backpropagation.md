# Backpropagation

The dense layer computes:

```text
Z = XW + b
A = activation(Z)
```

During the backward pass it receives `dA` from the next layer. If the output loss
already provides a pre-activation delta, the output layer can skip the local activation
derivative. This is used for:

- sigmoid + binary cross-entropy
- softmax + categorical cross-entropy

For hidden layers, the local derivative is applied:

```text
dZ = dA * activation'(Z)
dW = X.T @ dZ
db = sum(dZ)
dX = dZ @ W.T
```

The implementation stores forward-pass caches only during training, so inference remains
a pure forward computation.
