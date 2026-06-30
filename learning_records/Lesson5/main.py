"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os

import torch
from torch import nn
from pathlib import Path
import torchvision
from tqdm.auto import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchinfo import summary

NUM_WORKERS = 0

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

train_dir = image_path / "train"
test_dir = image_path / "test"

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names

manual_transforms = transforms.Compose([
  transforms.Resize((224,224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]) 
])

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
auto_transforms = weights.transforms()

train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,test_dir=test_dir,transform=auto_transforms,batch_size=32)
model = torchvision.models.efficientnet_b0(weights=weights)

for param in model.features.parameters():
    param.requires_grad = False

torch.manual_seed(42)
torch.cuda.manual_seed(42)

output_shape = len(class_names)

model.classifier = torch.nn.Sequential(
   torch.nn.Dropout(p=0.2,inplace=True),
   torch.nn.Linear(in_features=1280,
                   out_features=output_shape,
                   bias=True)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.001)

def train_step(model: nn.Module,
                   dataset: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module,
                   optimizer: torch.optim.Optimizer) : 
        
        model.train()

        train_loss, train_acc = 0, 0

        for batch, (X,y) in enumerate(dataset):
            
            y_pred = model(X)

            loss = loss_fn(y_pred,y)

            train_loss += loss

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            y_pred_class = torch.eq(torch.argmax(torch.softmax(y_pred,dim=1),dim=1),y).sum().item() 
            train_acc += y_pred_class / len(y_pred)

        # Adjust metrics to get average loss and accuracy per batch 
        train_loss = train_loss / len(dataset)
        train_acc = train_acc / len(dataset)
        return train_loss, train_acc
    
def test_loop(model: nn.Module,
                dataset: torch.utils.data.DataLoader,
                loss_fn: nn.Module) : 
    
    model.eval()

    test_loss, test_acc = 0, 0

    for batch, (X,y) in enumerate(dataset):

        test_pred = model(X)

        loss = loss_fn(test_pred, y)

        test_loss += loss

        y_label = torch.argmax(torch.softmax(test_pred,dim=1),dim=1)
        test_acc += torch.eq(y_label,y).sum().item() / len(test_pred)

    test_loss /= len(dataset)
    test_acc /= len(dataset)

    return test_loss, test_acc
    
def train(model: torch.nn.Module,
            train_dataloader: torch.utils.data.DataLoader,
            test_dataloader: torch.utils.data.DataLoader,
            loss_fn: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            epochs=5) : 
    
    results = {
        "Train_loss": [],
        "Train_acc" : [],
        "Test_loss" : [],
        "Test_acc" : []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model,train_dataloader,loss_fn,optimizer)
        test_loss, test_acc = test_loop(model,test_dataloader,loss_fn)

        results["Train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["Train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["Test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["Test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)


    return results

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Setup training and save the results
results = train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=5)

print(results)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")