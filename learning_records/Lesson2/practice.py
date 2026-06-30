# このファイルでは①から順番にコードを書いているから
# もし見る人がいたらこの番号順で見てください

# ①まずは必要なライブラリのインポートから
import torch

# torchとこのnnはニューラルネットワークや畳み込みなどの作業をする時に必要
from torch import nn

# データの表示
import matplotlib.pyplot as plt

# torch経由でMNISTデータを入れる
from torchvision import datasets

# ToTensorは名前の通り、データをテンソル型に変える
# これが必要な理由としては、最初このMNISTのデータを入れるときは
# テンソル型ではなく画像データを扱う別の型になっているからこれを使う
from torchvision.transforms import ToTensor

# DataLoaderは塊としてあるデータを一括に処理するのではなく
# いくつかに分けてモデルに学習をさせる
from torch.utils.data import DataLoader

# 主に画像を取り扱うもの
import torchvision

# ここ手順④だから読み進める時は注意！！！
# ④ニューラルネットワーク（全結合と畳み込み）を用いてモデルを作っていく
# モデル名は適当
# ここのnn.Moduleを入れることによってこのクラスを継承している
# このnn.Moduleというのはいわばレゴを作るときのブロックのようなもので
# ニューラルネットワークを構築するときの基盤となっている
# このMNistまでの過程も含めて、わかりやすく解説してくれているのがこのサイトなので読むのが一番いいかも
# https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html#refactor-using-nn-module
# 端的にまとめると、nn.Moduleを受け継いだクラスを作ることによって、いちいちフィールドにパラメータを保持する必要がなく
# またtrain_loopを実行するときに手動で重みやバイアスを更新する必要がなくなる
class DigitClassifier(nn.Module):

    # 二つここには書かないといけないことがある
    # それはコンストラクタ（重みやバイアスなどを記録する場所）
    # そしてforward（計算式）

    def __init__(self, in_features, out_features, hidden_features):

        super().__init__()

        # Sequentialは言い換えるとパイプラインのようなもので
        # ニューラルネットーワークの中の層やアクティベーションの設定を
        # 一括でできるようになっている
        # とりあえずこっちの方では全結合の形で実装をする
        # もう一つクラスを作るが、そっちの方では畳み込みで実装をする
        # ======================== ここでは全結合を用いてモデルを作る
        # ちなみにニューラルネットワークに関してはここら辺を見たらわかりやすいかも
        # https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.34630&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false 
        # self.layer = nn.Sequential(

            # 流れとしては最初は入力そう、真ん中には隠れそうがある
            # 活性化関数としてはReLU関数を用いる(負の値を0に、正の値はそのまま)
            # これによって表現の幅が広がる
            # 気になる人はReLU()をコメントアウトしてからやってみるといいかも
            # 後そうだ、画像データは[1,28,28]みたいになっているから、これじゃだめ
            # じゃあどうしたらいいかというと最初に画像を平坦化する(28行28列のデータを1行784列にするイメージ)
            # nn.Flatten(),
            # nn.Linear(in_features=in_features, out_features=hidden_features),
            # nn.ReLU(),
            # nn.Linear(in_features=hidden_features, out_features=out_features)

        # )

        # ======================== ここからは畳み込みを使ってモデルを作る

        self.block1 = nn.Sequential(
            # 畳み込みのイメージとしては特定のピクセルの周りkernl_sizeの場所を見て、
            # そこから特徴を取り出していくようなイメージ
            # これで例えば縦のパターンとか横のパターンをうまく見つけていくイメージかな
            # イメージはこれで掴みやすいかも
            # https://poloclub.github.io/cnn-explainer/
            nn.Conv2d(in_channels=in_features, 
                      out_channels=hidden_features, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_features, 
                      out_channels=hidden_features,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_features*7*7, 
                      out_features=out_features)
        )


    # ここでの計算はニューラルネットワークに画像を入れる。
    def forward(self, x):
        x = self.block1(x)
        print("==block1==")
        print(x.shape)
        x = self.block_2(x)
        print("==block2==")
        print(x.shape)
        x = self.classifier(x)
        print("==classi")
        print(x.shape)
        return x



# ②画像処理に使う画像のデータをダウンロードしていく
# データの読み込み
# 引数の部分は省略します
train_data = datasets.MNIST(root='data',
                            train=True,
                            transform=ToTensor(),
                            download=True)

test_data = datasets.MNIST(root='data',
                           train=False,
                           transform=ToTensor(),
                           download=True)


# ③データローダーを用いて学習の準備
# 60000個のデータを一括に処理するのは効率が悪い（パフォーマンスが悪い)
# データをいくつかに分けて、そこから学習させる
# batch_size = 幾つにデータを分けるか
# 今回は32（ネットによると適切らしい）を使う
# shuffleデータをランダムに混ぜる
# ステップ④は上に戻るから注意（次はクラスを作っていく
BATCH_SIZE = 32
train_dataloader = DataLoader(train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
# テストデータにも同じことをする
test_dataloader = DataLoader(test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

# モデルはクラスなのでインスタンスに
model = DigitClassifier(in_features=1, out_features=10, hidden_features=10)

# そしたら損失関数とそれを最適化するoptimizerを使う
# 今回はlog_soft_max()と最尤推定で誤差を計算する
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

# 同じデータを何周して学習するかを決める
# 少ないとunder_fitting、逆に多すぎるとover_fittingになるから調整は必要
# 多すぎると時間かかるからepoch=10で

epochs = 10

for epoch in range(epochs):

    for X,y in train_dataloader:
        model.train()

        y_pred = model(X)
        # 誤差
        loss = loss_fn(y_pred,y)
        # 異なるセットを学習するごとにoptimizerはリセットしとかないといけない
        # そうじゃないと前のデータの勾配が残ったまま学習してしまう
        optimizer.zero_grad()
        # 連鎖率を用いて誤差逆伝播法で微分していく
        loss.backward()

        # パラメータの更新
        optimizer.step()

# とりあえず結果を見ていく

with torch.inference_mode():
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    for imgs, labels in test_dataloader:
        y_logits = model(imgs)
        loss = loss_fn(y_logits,labels)
        y_pred = y_logits.argmax(dim=1)

        correct += (y_pred == labels).sum().item()
        total += imgs.size(0)
        loss_sum += loss.item() * imgs.size(0)
    
    avg_loss = loss_sum / total
    acc = correct / total * 100
    print(f"Current loss: {avg_loss:.3f}, accuracy: {acc:.3f}%")
