import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleMLP


def get_dataloaders(batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    total_loss = 0
    for batch, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def test(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
    accuracy = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy


def main():
    parser = argparse.ArgumentParser(description="Train a simple MLP on MNIST")
    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--hidden-size', type=int, default=128, help='number of hidden units')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_dataloaders(args.batch_size)

    model = SimpleMLP(hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, accuracy = test(model, test_loader, criterion, device)
        print(f"Epoch {epoch}: train loss={train_loss:.4f}, test loss={test_loss:.4f}, accuracy={accuracy*100:.2f}%")

    torch.save(model.state_dict(), 'mnist_mlp.pth')
    print('Model saved to mnist_mlp.pth')


if __name__ == '__main__':
    main()
