import torch
from torch import nn
from torch.utils.data import DataLoader, dataset
import torchvision
from torchvision.datasets import ImageFolder

import pandas as pd

from torchvision.transforms import transforms

from pathlib import Path

from torchinfo import summary

from tqdm.auto import tqdm

# 複数のモデルをテストする時に役立つみたい
# 実際にはこれ以外にもあって、"machine learning experiment tracking"と調べると色々あるらしい
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import os

# ここにSummaryWriterを作ってくれる関数を用意して作成する

def save_model(model: nn.Module,
               target_dir: str,
               model_name) :
    
    path = Path(target_dir)
    path.mkdir(parents=True, exist_ok=True)

    path_name = model_name
    save_path = path / path_name

    print("Now saving...")
    torch.save(obj=model.state_dict(), f=save_path)
    print("Completed!!")
    

def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter:
    
    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryInfo Writer. Saving to {log_dir}")
    return SummaryWriter(log_dir=log_dir)

writer = example_writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb0",
                               extra="5_epochs")
# データを読み込んで、読み込んだ訓練とテストデータと、最後に分類する種類の名前を返す
def DataLoading(train_dir : str,
                test_dir  : str,
                transform : transforms.Compose,
                batch_size = 10, 
                num_workers = 0) : 
    

    train_dataset = ImageFolder(root=train_dir,
                                transform=transform,)
    
    test_dataset  = ImageFolder(root=test_dir, 
                                transform=transform)
    
    class_names = train_dataset.classes

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True)
    
    test_dataloader  = DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers)
    
    return train_dataloader, test_dataloader, class_names

# ==================== Train and Test Loop functions starts ====================

def train_step(model: nn.Module,
          dataset: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer):

    model.train()

    total_loss, total_acc = 0, 0
    for batch, (X,y) in enumerate(dataset):

        y_pred = model(X)

        y_loss = loss_fn(y_pred,y)

        total_loss += y_loss

        y_prob = torch.argmax(y_pred, dim=1)
        total_acc += torch.eq(y_prob,y).sum().item() / len(y_prob) 

        optimizer.zero_grad()

        y_loss.backward()

        optimizer.step()

    total_loss /= len(dataset)
    total_acc /= len(dataset)

    return total_loss, total_acc

def test_step(model: nn.Module,
         dataset: torch.utils.data.DataLoader,
         loss_fn: torch.nn.Module):
    
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


def train(model: nn.Module,
          train_data: torch.utils.data.DataLoader,
          test_data : torch.utils.data.DataLoader,
          loss_fn : torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs = 5,
          writer : torch.utils.tensorboard.writer.SummaryWriter = None) :

    result = {
        "Train_loss": [],
        "Train_acc" : [],
        "Test_loss" : [],
        "Test_acc"  : []
    }

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = train_step(model=model,dataset=train_data,loss_fn=loss_fn,optimizer=optimizer)
        test_loss , test_acc  = test_step(model=model,dataset=test_data,loss_fn=loss_fn)

        result["Train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        result["Train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        result["Test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        result["Test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(main_tag="Loss", 
                               tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", 
                               tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc}, 
                               global_step=epoch)

            # Close the writer
            writer.close()
        else:
            pass
    
    return result

# ==================== Train and Test Loop functions ends   ====================

# Pathの指定
path = Path("./data/")
# 複数のモデルを使って検証をしたい
# 10% or 20% of food data
# the number of epoch
# the model
train_dir1 = path / "pizza_steak_sushi" / "train"
train_dir2 = path / "pizza_steak_sushi_20_percent" / "train"

test_dir  = path / "pizza_steak_sushi" / "test"

# 転移学習をするのでそのモデルの重みや変換方法をtransformとして扱う.
# About EfficientNet_B0_Weights
# https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # values per colour channel [red, green, blue]
                                 std=[0.229, 0.224, 0.225]) # values per colour channel [red, green, blue]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
])

# ここでは処理（順番）をわかりやすくしたいので、データのローディング作業は関数に入れてすっきりさせます
# やってることはそんなに難しいことじゃない
train_dataloader_10_percent, test_dataloader, class_names = DataLoading(train_dir1,test_dir,transform)
train_dataloader_20_percent, test_dataloader, class_names = DataLoading(train_dir2,test_dir,transform)
# 核となるモデルをインポートする（なかったらダウンロードされるのかな？）
model = torchvision.models.efficientnet_b0(weights=weights)

weights2 = torchvision.models.EfficientNet_B2_Weights.DEFAULT
model2 = torchvision.models.efficientnet_b2(weights=weights2)

# result = summary(model=model2,
#                  input_size=(32,3,224,224),
#                  col_names=["input_size", "output_size","num_params","trainable"],
#                  col_width = 20,
#                  row_settings=["var_names"])
# print(result)

# 核となる部分の重みとかはすでに最適となっているはずなので、
# そこの部分は更新したくない。ということでrequres_gradをFalseにして勾配は残さない
for feature in model.features.parameters():
    feature.requires_grad = False

for feature in model.features.parameters():
    feature.requires_grad = False

# torchinfoに関するドキュメントは一応ここらしい
# https://github.com/TylerYep/torchinfo/blob/main/torchinfo/torchinfo.py
# 詳しい説明はこのリンクで見てください

# result = summary(model=model,
#                  input_size=(32,3,224,224),
#                  col_names=["input_size", "output_size","num_params","trainable"],
#                  col_width = 20,
#                  row_settings=["var_names"])
# print(result)

# 今回最後分別するのが1000種類のデータではなく
# 寿司、ステーキ、ピザの3種類しかないので、そこの部分を変更していく
# ここの変更するために上のようにsummaryとすると結構便利
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280,
              out_features=len(class_names),
              bias=True)
)

model2.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1408,
              out_features=len(class_names),
              bias=True)
)

num_epochs = [5, 10]

models = ["effnetb0", "effnetb2"]

# 2. Keep track of experiment numbers
experiment_number = 0

train_dataloaders = {"data_10_percent": train_dataloader_10_percent,
                     "data_20_percent": train_dataloader_20_percent}

for dataloader_name, train_dataloader in train_dataloaders.items():

    for epochs in num_epochs:

        for model_name in models:

            experiment_number += 1

            print(f"[INFO] Experiment number: {experiment_number}")
            print(f"[INFO] Model: {model_name}")
            print(f"[INFO] DataLoader: {dataloader_name}")
            print(f"[INFO] Number of epochs: {epochs}") 

            if model_name == "effnetb0":
                current_model = model
            else :
                current_model = model2

             # log_Softmax + 最尤推定
            # log_Softmax = ei - logΣ e^ej
            # NLL = -Σlog(py) (y=正解ラベルでpyはその確率、大きければ答えは0に近づき、全然違うかったらでかい数になる。そのため-をつけて処理しやすいようにしてる。)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(params=model.parameters(),lr=0.001)

            train(model=current_model,train_data=train_dataloader,test_data=test_dataloader,loss_fn=loss_fn,optimizer=optimizer,epochs=epochs,writer=create_writer(experiment_name=dataloader_name,
                                                                                                                                                                   model_name=model_name,
                                                                                                                                                                   extra=f"{epochs}_epochs"))

save_filepath = f"07_{model_name}_{dataloader_name}_{epochs}_epochs.pth"
save_model(model=model,
            target_dir="models",
            model_name=save_filepath)
print("-"*50 + "\n")
