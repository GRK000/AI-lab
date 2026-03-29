from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError as error:
    raise SystemExit(
        "PyTorch is not installed. Install it with: pip install torch"
    ) from error

from common import own_network_result, xor_dataset


class TorchXORNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


def main() -> None:
    torch.manual_seed(21)

    X, y = xor_dataset()
    inputs = torch.tensor(X, dtype=torch.float32)
    targets = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

    model = TorchXORNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.BCELoss()

    for _ in range(3000):
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predicted = (model(inputs) >= 0.5).to(dtype=torch.int64).view(-1)
        accuracy = float((predicted == torch.tensor(y)).float().mean().item())

    own_result = own_network_result()

    print("=== XOR comparison against PyTorch ===")
    print(f"{own_result.model_name}: score={own_result.train_score:.4f}")
    print(f"PyTorch network: score={accuracy:.4f}")


if __name__ == "__main__":
    main()
