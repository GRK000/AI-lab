# Optimizers

Optimizers implement one narrow contract:

```python
update(parameter_key, parameter, gradient) -> updated_parameter
```

The `parameter_key` lets optimizers keep independent state for every weight and bias
array. This supports:

- SGD
- Momentum
- RMSprop
- Adadelta
- Adam
- AdamW
- Adamax

The training loop calls `begin_step()` once per batch, then each trainable layer asks
the optimizer to update its parameters.
