"""
mrdbrouke氏のZero to Mastery Learn PyTorch for Deep Learningを一通り見て勉強したので、その復習用としてもう一度最初からモデルを作ります
最終的なゴールはKaggleで見つけた Flowers Dataset を用いて、画像から5つの種類のうちどの花なのかを判別できるモデルを作ること
Link: https://www.kaggle.com/datasets/imsparsh/flowers-dataset/data
主にPyTorchを使ってモデルを作っていきます。また、できるだけコメントアウトで説明も加えていこうと思います。
"""

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
from torchvision.transforms import transforms

# Python IMage Libraryの略。パスから画像を読み取ってくれる
from PIL import Image
# ファイルのパスを読み取る時に便利
from pathlib import Path
# 型を明記するようにします
from typing import Dict, List, Tuple

# こっちの方に色々関数をまとめます
from func import create_dataloader, train_loop, create_writer

from torch.utils.tensorboard import SummaryWriter

from torchinfo import summary

from tqdm.auto import tqdm

# モデルを作る時は必ずnn.Moduleを継承する
class MyClassficationModel(nn.Module) :

    # モデルを作る時に必要なのは二つだけ
    # コンストラクタとforward（入力された画像とかを学習させる役割）

    def __init__(self, in_features, hidden_features, out_features):

        super().__init__()

        # とりあえずブロックごとに分けていく
        # 今回の畳み込みとは少し違うが、ニューラルネットワークのイメージを持つ時は下のやつがおすすめ
        # https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.64822&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false
        # 畳み込みCNNに関しては下がわかりやすい
        # https://poloclub.github.io/cnn-explainer/
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features,
                      out_channels=hidden_features,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_features,
                      out_channels=hidden_features,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_features,
                      out_channels=hidden_features,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_features,
                      out_channels=hidden_features,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_features*64*64,out_features=out_features)
        )
    

    def forward(self, x):
        
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classify(x)
        return x

# とりあえず学習させたいデータのパスを用意する
train_path = Path("data/train")
test_path = Path("data/test")

# グラフを示してその比較とかもしてみたいから、複数のモデルを作ろうと思います（色々なトランスフォーマーが必要）
# ここでいうトランスフォームが何を言っているかというと、今ある画像データを学習できるような適切な形に変換をしていく
# 例えばToTensorというのは元々画像はPILの形になっているから、これを学習させるためにはTensor(テンソル)の形に変えないといけない（まあベクトルの集まりみたいなもの？）
# それがToTensor()で簡単に変換できる
# 他にも転移学習をするときインプットののサイズなどに指定があることもあるので、画像サイズも適切な形に変えなければならない
# そこでResize(H,W)みたいな感じで画像サイズの変換をしないといけない
# そこで、今回はとりあえずまずは一つは自分でモデルを作ってみて、他二つを転移学習を用いたモデルにしていこうと思います
# 転移学習に使うモデルはEfficientNetとVisionTransformerを使っていこうと思います
# 参考1: https://arxiv.org/abs/1905.11946 (EfficientNet)
# 参考2: https://arxiv.org/abs/2010.11929 (VisionTransformer)
# すごい正直なことを言うとあまり中身を理解できませんでした... 数式怖い...
# とりあえず三つ作っていく

# mode1: 自分で作るやつ。シンプルに畳み込みとReLU関数を作ってやってみる
# とりあえずモデルは上に作る
my_model = MyClassficationModel(in_features=3, hidden_features=30, out_features=5)

# 変換するやつ
transformer = torchvision.transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

# データセット（学習用）を作成
train_dataloader, test_dataloader, class_names = create_dataloader(train_path=train_path,test_path=test_path,transformer=transformer,batch_size=32)

# epochs = 5

# for epoch in range(epochs): 

#     train_loss, train_acc = train_loop(model=my_model,dataset=train_dataloader,loss_fn=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam(params=my_model.parameters(), lr=0.001))

#     print(f"Epoch: {epoch} | Loss: {train_loss} | Acc: {train_acc}")

# とりあえずmodel2まで作ってみる
# model2としてはとりあえずEfficientNetb2を使ってやってみる
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
transformer = weights.transforms()
effnetb0 = torchvision.models.efficientnet_b0(weights=weights)

# info = summary(model=effnetb0,input_size=(32,3,256,256),col_names=["input_size","output_size","num_params","trainable"],row_settings=["var_names"])

for param in effnetb0.parameters():
    param.requires_grad = False

effnetb0.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(
        in_features=1280,
        out_features=3
    )
)

train_loss, train_acc = train_loop(model=my_model,dataset=train_dataloader,loss_fn=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam(params=my_model.parameters(), lr=0.001),epochs=5, writer=create_writer("Test","AA"))
train_loss_2, train_acc_2 = train_loop(model=effnetb0,dataset=train_dataloader,loss_fn=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam(params=effnetb0.parameters(), lr=0.001),epochs=5, writer=create_writer("Test","FA"))

