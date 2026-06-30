# ここのファイルはmain.pyのところのサブ的な役割
# 主にカスタムデータセットの作り方をこっちの方に書いていこうと思う。
# 理由はシンプルにややこしそうだから。

# os = ファイル操作に使える
# Pathもそのままの通りファイルのパスのこと
import os
import pandas as pd
from pathlib import Path
import pathlib

# みんな大好きデータセットを作れたりするし、ニューラルネットワークも作れるtorch
import torch
from torch import nn

# ランダム
import random

# PIL stands for "Python Imaging Library" 要するに画像表示とかできる
from PIL import Image
# カスタムデータセットを作るときはDatasetを継承して作成する
# バッチごとにデータを読み込んだりする際はDataLoader
from torch.utils.data import Dataset, DataLoader
# tensorに変換するときなどに使う
from torchvision import transforms
# 型を明記する際に使えるみたい
# Pythonって基本的に型がなくても動くけど
# こうすることによって将来的に起こるようなミスを防げる
from typing import Tuple, Dict, List

# 画像表示や表を表示するやつ
import matplotlib.pyplot as plt

# デバッグ用？
# 詳細の情報が見れる。インプットのデータのサイズとか、パラメータの数とかそういうの
import torchinfo

from tqdm.auto import tqdm

# モデルを作っていく。
# 参考にするのはVGGの簡単なバージョン
# VGG自体は有名なモデルのアーキテクチャで、効率がいいらしい
class TinyVGG(nn.Module):

        def __init__(self, in_channels, out_channels, hidden_channels) : 

            super().__init__()

            self.block_1 = nn.Sequential(
                nn.Conv2d(
                    # 今回は色付き(R,G,B)だから入力は3
                    in_channels=in_channels,
                    # ここで言えばHidden sizeになるかな
                    out_channels=hidden_channels,
                    # カーネルサイズっていうのは畳み込みをする時、n*nの四角形を見るかというのを設定する
                    # n=3だったら3*3の正方形を動かしながら特徴を捉えていく
                    kernel_size=3,
                    # paddingっていうのは周りに余分な0のピクセルを加えることによって端っこの特徴をより捉えるようにする
                    padding=1
                ),

                # 活性化関数ReLU()を使って、負の値を0に変換する（強い特徴をしっかりと捉えそこに注目する）
                nn.ReLU(),  
                nn.Conv2d(
                    # 今回は色付き(R,G,B)だから入力は3
                    in_channels=hidden_channels,
                    # ここで言えばHidden sizeになるかな
                    out_channels=hidden_channels,
                    # カーネルサイズっていうのは畳み込みをする時、n*nの四角形を見るかというのを設定する
                    # n=3だったら3*3の正方形を動かしながら特徴を捉えていく
                    kernel_size=3,
                    # paddingっていうのは周りに余分な0のピクセルを加えることによって端っこの特徴をより捉えるようにする
                    padding=1
                ),
                nn.ReLU(),
                # ここでは上のConv2dと違って学習はしない状態で、2x2のピクセルの中で一番特徴が強いところだけを、
                # ピックアップし、それをその周辺のピクセルの代表にするみたいな感じかな？
                nn.MaxPool2d(2)
            ) 

            self.block_2 = nn.Sequential(
                nn.Conv2d(
                    # 今回は色付き(R,G,B)だから入力は3
                    in_channels=hidden_channels,
                    # ここで言えばHidden sizeになるかな
                    out_channels=hidden_channels,
                    # カーネルサイズっていうのは畳み込みをする時、n*nの四角形を見るかというのを設定する
                    # n=3だったら3*3の正方形を動かしながら特徴を捉えていく
                    kernel_size=3,
                    # paddingっていうのは周りに余分な0のピクセルを加えることによって端っこの特徴をより捉えるようにする
                    padding=1
                ),

                # 活性化関数ReLU()を使って、負の値を0に変換する（強い特徴をしっかりと捉えそこに注目する）
                nn.ReLU(),  
                nn.Conv2d(
                    # 今回は色付き(R,G,B)だから入力は3
                    in_channels=hidden_channels,
                    # ここで言えばHidden sizeになるかな
                    out_channels=hidden_channels,
                    # カーネルサイズっていうのは畳み込みをする時、n*nの四角形を見るかというのを設定する
                    # n=3だったら3*3の正方形を動かしながら特徴を捉えていく
                    kernel_size=3,
                    # paddingっていうのは周りに余分な0のピクセルを加えることによって端っこの特徴をより捉えるようにする
                    padding=1
                ),
                nn.ReLU(),
                # ここでは上のConv2dと違って学習はしない状態で、2x2のピクセルの中で一番特徴が強いところだけを、
                # ピックアップし、それをその周辺のピクセルの代表にするみたいな感じかな？
                nn.MaxPool2d(2)
            )

            self.classify = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=hidden_channels*16*16,out_features=out_channels)
            )
        
        def forward(self, x):
            x = self.block_1(x)
            # print(x)
            x = self.block_2(x)
            # print(x)
            x = self.classify(x)
            return x

# カスタムデータセットを作成する
# まず第一に親のDatasetを継承する
class ImageFolderCustom(Dataset):

    # この中で必ず作らないといけないメソッドは3つ
    # __init__(コンストラクタ)、__len__(サイズ)、 __getitem__(要素の取得)

    # targ_dir = クラス分類されてるパス、transformは変換機みたいなもの
    def __init__(self, targ_dir: str, transform=None) -> None : 
        # 画像の取得
        # *クラスごとに分けられている/*全ての画像のパスを取得する
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        # どのようにデータを調整するのかの指定
        self.transform = transform
        # ここで画像データの種類と数字で紐付ける
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 画像の表示
    def load_image(self, index: int) -> Image.Image : 
        image_path = self.paths[index]
        return Image.open(image_path)
    
    # データの大きさを取得  
    def __len__(self) -> int :
        return len(self.paths)
    
    # 特定の要素の取得（クラスの画像とそのラベル）
    def __getitem__(self, index: int) -> Tuple[torch.Tensor,int] : 
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx

# test or train　フォルダの中ににあるフォルダの中には特定の画像の集合となっている
# そして、osscandirで指定したフォルダの中身を全てチェック。それをソート
# そして、もしファイルが存在すればクラス名と適当なインデックス番号を紐付けしてそれを返すようになっている
# Tuple[List[str], Dict[str,int]]というのは戻り値の型
def find_classes(directory : str) -> Tuple[List[str], Dict[str,int]]:

    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Couldn't find any in {directory}")
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def main():

    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"

    random.seed(42)

    image_path_list = list(image_path.glob("*/*/*.jpg"))

    # 画像を特定の形に変換するもの
    # composeで組み立てみたいな感じかな？
    # Resize(size=(x,y))で幅x高さyの画像に縮小
    # 次に画像を反転させることもある(バリエーションを増やすためかな？)
    # そして最後に画像をTensorに変える。これはもちろん処理をするため
    train_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor()
    ])


    test_transform = transforms.Compose([
        transforms.Resize(size=(64,64)),
        transforms.ToTensor()
    ])

    train_dir = "./data/pizza_steak_sushi/train"
    test_dir = "./data/pizza_steak_sushi/test"

    target_dir = train_dir

    train_data_custom = ImageFolderCustom(train_dir,transform=train_transform)
    test_data_custom = ImageFolderCustom(test_dir,transform=test_transform)


    # def display_random_images(dataset: torch.utils.data.Dataset,
    #                         classes: List[str]=None,
    #                         n: int=10,
    #                         display_shape: bool=True,
    #                         seed: int= None):
        
    #     if n > 10:
    #         n = 10
    #         display_shape = False
    #         print("N cant't be more than 11. We set it to 10 instead")

    #     if seed:
    #         random.seed(seed)

    #     random_sample_idx = random.sample(range(len(dataset)), k=n)

    #     plt.figure(figsize=(16,8))

    #     for i, targ_sample in enumerate(random_sample_idx):
    #         targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

    #         targ_image_adjust = targ_image.permute(1,2,0)

    #         plt.subplot(1,n,i+1)
    #         plt.imshow(targ_image_adjust)

    #         if classes:
    #             title = f"Class: {classes[targ_label]}"
    #         plt.title(title)
    #         plt.axis(False)
        
    #     plt.show()

    # display_random_images(train_data_custom, 
    #                       n=10, 
    #                       classes=train_data_custom.classes,
    #                       seed=None)

    num = os.cpu_count()

    train_datafolder_custom = DataLoader(dataset=train_data_custom,
                                        batch_size=10,
                                        num_workers=num-1,
                                        shuffle=True)

    test_datafolder_custom = DataLoader(dataset=test_data_custom,
                                        batch_size=10,
                                        num_workers=num-1)

    torch.manual_seed(42)
    model_0 = TinyVGG(in_channels=3, hidden_channels=30, out_channels=len(train_data_custom.classes))

    # img_batch, label_batch = next(iter(train_datafolder_custom))

    # img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]

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
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(),lr=0.001)

    model_0_results = train(model_0,train_datafolder_custom,test_datafolder_custom,loss_fn,optimizer,epochs=20)

    df = pd.DataFrame(model_0_results)

    print(df)

if __name__=="__main__":
    main()