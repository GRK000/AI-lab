from __future__ import annotations

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import Perceptron as SkPerceptron
    from sklearn.neural_network import MLPClassifier
except ModuleNotFoundError as error:
    raise SystemExit(
        "scikit-learn is not installed. Install it with: pip install scikit-learn"
    ) from error

from common import (
    binary_dataset,
    own_network_result,
    own_neuron_result,
    own_perceptron_result,
    xor_dataset,
)


def main() -> None:
    X_binary, y_binary = binary_dataset()
    X_xor, y_xor = xor_dataset()

    own_perceptron = own_perceptron_result()
    own_neuron = own_neuron_result()
    own_network = own_network_result()

    sklearn_perceptron = SkPerceptron(
        eta0=0.2,
        max_iter=1000,
        tol=1e-4,
        shuffle=True,
        random_state=7,
    )
    sklearn_perceptron.fit(X_binary, y_binary)

    logistic = LogisticRegression(random_state=7)
    logistic.fit(X_binary, y_binary)

    mlp = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="tanh",
        solver="adam",
        max_iter=4000,
        random_state=21,
    )
    mlp.fit(X_xor, y_xor)

    print("=== Binary comparison ===")
    print(f"{own_perceptron.model_name}: score={own_perceptron.train_score:.4f}")
    print(f"sklearn Perceptron: score={sklearn_perceptron.score(X_binary, y_binary):.4f}")
    print(f"{own_neuron.model_name}: score={own_neuron.train_score:.4f}")
    print(f"LogisticRegression: score={logistic.score(X_binary, y_binary):.4f}")

    print("\n=== XOR comparison ===")
    print(f"{own_network.model_name}: score={own_network.train_score:.4f}")
    print(f"MLPClassifier: score={mlp.score(X_xor, y_xor):.4f}")


if __name__ == "__main__":
    main()
