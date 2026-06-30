# モデルを作るときにメインとなるライブラリ
# 主な用途としてはモデルの基礎を作る、損失関数を作る、最適化関数を作る（SGDとかAdamとか）、モデルを用いて学習させることができるなどなど...
import torch
# 楽だからこうしてます。nnはニューラルネットワークを作る時に役立ちます。
from torch import nn
# このDatasetとDataLoaderは主に、学習するファイルを読み込むために使うもの。あとラベル付けをしてくる(class_names)
# Datasetはデータを適切な形に変換するときなどに使う。そして、DataLoaderはデータをバッチごとに分けたり効率よく学習させる設定をする
from torch.utils.data import Dataset, DataLoader

# torchvisionはいろいろなデータセットが入っていたり、また転移学習などにも使える様々なデータセットが入っている。
import torchvision
# 上でDatasetを使うと書いているが、実際今回使うのはこっち。画像に適したDatasetだと思ったらいい
from torchvision.datasets import ImageFolder

# Python IMage Libraryの略。パスから画像を読み取ってくれる
from PIL import Image
# ファイルのパスを読み取る時に便利
from pathlib import Path
# 型を明記するようにします
from typing import Dict, List, Tuple

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import os

from tqdm.auto import tqdm

# データローダの関数を作っときます
def create_dataloader(train_path: str,
                      test_path : str,
                      transformer,
                      batch_size : int
                      ) :
    
    train_dataset = ImageFolder(root=train_path,transform=transformer)

    test_dataset  = ImageFolder(root=test_path,transform=transformer)

    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    class_names = train_dataset.classes

    return train_dataloader, test_dataloader, class_names


def train_loop(model: nn.Module,
               dataset: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               epochs: int,
               writer) :
    
    model.train()

    result = {
        "Train_loss": [],
        "Train_acc" : []
    }

    for epoch in tqdm(range(epochs)):

        total_loss, total_acc = 0, 0

        for (X,y) in dataset: 

            y_pred = model(X)

            loss = loss_fn(y_pred,y)
            total_loss += loss

            y_prob = torch.argmax(torch.softmax(y_pred,dim=1), dim=1)

            total_acc += torch.eq(y_prob,y).sum().item() / len(X)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        total_loss /= len(dataset)
        total_acc /= len(dataset)

        result["Train_loss"].append(total_loss.item() if isinstance(total_loss,torch.Tensor) else total_loss)
        result["Train_acc"].append(total_acc.item() if isinstance(total_acc,torch.Tensor) else total_loss)

        if writer:

            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={"train_loss": total_loss},
                               global_step=epoch)
            
            writer.add_scalars(main_tag="Acc",
                               tag_scalar_dict={"train_acc" : total_acc},
                               global_step=epoch)
            
            writer.close()

    return total_loss, total_acc


def create_writer(experiment_name : str,
                  model_name: str,
                  extra=None) : 
    
    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        log_dir = os.path.join("runs",timestamp,experiment_name,model_name,extra)
    else :
        log_dir = os.path.join("runs",timestamp,experiment_name,model_name)
    
    return SummaryWriter(log_dir=log_dir)
    