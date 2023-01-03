from vision_models import PMDatasets
from torchvision.transforms import ToTensor, Compose, Resize
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import torch.nn as nn
import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
num_classes = 3

# todo: 增加一些代表性的数据增强的 transforms

transforms = Compose([ToTensor(), Resize((224, 224))])

training_dataset = PMDatasets(data_path='data/imagenet', data_type='train', transforms=transforms)
testing_dataset = PMDatasets(data_path='data/imagenet', data_type='test', transforms=transforms)

training_dataloader = DataLoader(training_dataset, batch_size=batch_size,shuffle=True)
testing_dataloader = DataLoader(training_dataset, batch_size=batch_size,shuffle=False)


## model defining
model_ft = resnet50(pretrained=True)

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)
model_ft = model_ft.to(device)

# todo: vision-transformer  swin-transformer VGG18 Resnet34

## model setting
critern = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.parameters(),lr=0.0001)
model_ft.train()

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def model_test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(training_dataloader, model_ft, critern, optimizer)
    model_test(training_dataloader, model_ft, critern)
    torch.save(model_ft.state_dict(), 'checkpoints/model'+str(t)+'.pth')
    print("Saved PyTorch Model State to model.pth")

print("Done!")