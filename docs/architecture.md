# Architecture

The project is split into a small reusable core and executable demos.

## Core

- `neural_core.activations`: activation functions and derivatives.
- `neural_core.layers`: trainable and non-trainable layer blocks.
- `neural_core.optimizers`: parameter update rules with state.
- `neural_core.network`: training loop, batching, losses, prediction and persistence.
- `neural_core.callbacks`: optional training lifecycle hooks.
- `neural_core.metrics`: evaluation helpers for classification and regression.

`NeuralNetwork` owns the training loop. Layers own forward/backward math. Optimizers
own parameter updates. This keeps backpropagation readable and avoids hiding model
behavior behind a framework.

## Demos And Benchmarks

- `examples/` contains user-facing examples.
- `comparisons/` compares the local implementation with external libraries.
- `benchmarks/` produces reproducible result tables for portfolio evidence.
