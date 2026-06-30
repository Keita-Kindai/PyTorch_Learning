# Lesson別ファイル整理・実行確認メモ

作成日: 2026-06-28  
確認環境: `/Users/keita/envs/def/bin/python`  
Python: 3.10.20  
PyTorch: 2.7.0  
torchvision: 0.22.0

このファイルは、Lessonごとの内容、実行確認結果、GitHubで見せるなら優先したいファイルをまとめたものです。

2026-06-30追記: 現在はリポジトリ見せ方を整理し、公開アプリを `my_project/`、学習履歴を `learning_records/`、整理前の元フォルダを `archive/original_workspace/` に分けています。このメモ内の `Lesson*` パスは、整理前の元フォルダ構成を前提にした記録です。

## 全体方針

このフォルダは、Lesson単位の学習ログとしては量が多いですが、就活用には以下の見せ方がよいです。

1. メイン成果物は `Lesson8/demos/my_project` の FoodVision Mini
2. 学習過程の証拠として `Lesson2/practice.py`、`Lesson4/custom.py`、`Lesson6/main.py`、`Lesson9_recap/main.py` を見せる
3. `basic.py`、`play.py`、`test.py`、`media10.ipynb` などLesson外の単発ファイルは、公開前に削除または `archive/` 行きでよい
4. `Lesson6/pytorch-deep-learning/` は教材リポジトリ本体に見えるため、自作コードとは分けた方がよい

## 実行確認の前提

実行確認は次の2段階で行いました。

- 全Pythonファイルに対する `py_compile`: 構文エラー確認
- Lesson直下・デモ配下の主要 `.py` を短時間実行: 実行時エラー確認

注意点:

- 学習が長いファイルは15から20秒でタイムアウトしています。これは必ずしもコード不具合ではなく、学習処理が始まっているためです。
- Gradioアプリはローカルの `7860-7959` ポートが埋まっていたため起動失敗しました。モデル読み込み自体は通っているため、ポート指定を変えれば起動できる可能性が高いです。
- 実行確認中に `Lesson6/runs/2026-06-28/` のTensorBoardログが生成されました。GitHub公開前には削除または `.gitignore` 対象にしてください。

## 構文チェック

結果: Lesson配下の `.py` は全て `py_compile` 成功。

対象には以下を含みます。

- `Lesson1` から `Lesson9_recap` のPythonファイル
- `Lesson8/demos/*/app.py`
- `Lesson8/demos/*/model.py`
- `Lesson6/pytorch-deep-learning/going_modular/going_modular/*.py`

## Lesson別まとめ

### Lesson1

テーマ: PyTorch基礎、表形式データ前処理、2次元分類、MNIST分類

| ファイル | 内容 | 見せ方 |
|---|---|---|
| `Lesson1/main.py` | pandas / sklearnで欠損値・カテゴリ変数を扱う練習。後半に学生就職/配置データの二値分類モデル案がコメントで残る | 補助的な学習ログ |
| `Lesson1/nn.py` | `make_circles` を使った2値分類。ReLU、BCEWithLogitsLoss、decision boundaryの可視化 | PyTorch分類入門として使える |
| `Lesson1/nn2.py` | `make_blobs` を使った4クラス分類。CrossEntropyLoss、softmax、decision boundary | 多クラス分類の基礎として使える |
| `Lesson1/mnist_digit.py` | MNIST手書き数字分類。全結合NN、DataLoader、Adam、評価 | 基礎モデル実装として使える |
| `Lesson1/helper_functions.py` | 可視化補助関数 | 補助ファイル |
| `Lesson1/test.py` | `numpy` importのみ | 削除候補 |

実行確認:

| ファイル | 結果 | メモ |
|---|---|---|
| `Lesson1/main.py` | OK | 表形式データの前処理結果を出力 |
| `Lesson1/nn.py` | OK | `Lesson1` をcwdにして `PYTHONPATH` にリポジトリルートを追加すると完走 |
| `Lesson1/nn2.py` | OK | `Lesson1` をcwdにして `PYTHONPATH` にリポジトリルートを追加すると完走 |
| `Lesson1/mnist_digit.py` | OK | リポジトリルートから実行するとMNIST既存データを利用して完走 |
| `Lesson1/helper_functions.py` | OK | import確認のみ |
| `Lesson1/test.py` | OK | 実質空ファイル |

注意:

- `python Lesson1/nn.py` をルートから実行すると、ルート直下に `helper_functions.py` がないためGitHubから取得しようとしてネットワークエラーになります。
- `python nn.py` を `Lesson1` 内から実行する場合は、`PYTHONPATH` にリポジトリルートを追加すると動きます。

### Lesson2

テーマ: FashionMNIST / MNIST、CNN、評価、混同行列、詳細コメント付き学習ログ

| ファイル | 内容 | 見せ方 |
|---|---|---|
| `Lesson2/main.py` | FashionMNIST分類。CNNモデル、保存済みモデル読み込み、予測、混同行列 | 完成度高めの学習ログ |
| `Lesson2/practice.py` | MNIST CNNをコメント付きで丁寧に実装。畳み込み、MaxPool、Flatten、評価まで | 学習過程を見せる筆頭候補 |
| `Lesson2/func.py` | train loop、eval、accuracy、prediction helper | 補助関数 |
| `Lesson2/helper_functions.py` | 可視化補助 | 補助関数 |
| `Lesson2/model/*.pth` | FashionMNIST/MNIST系の保存モデル | 成果物候補 |

実行確認:

| ファイル | 結果 | メモ |
|---|---|---|
| `Lesson2/main.py` | OK | ルートから実行で完走。混同行列表示はAgg環境のためGUI表示なし |
| `Lesson2/practice.py` | TIMEOUT | 学習は進行。20秒では完了せず |
| `Lesson2/func.py` | OK | import確認 |
| `Lesson2/helper_functions.py` | OK | import確認 |

注意:

- `Lesson2/main.py` はルートからの実行を想定しているように見えます。
- `Lesson2/practice.py` はコメントが多く、就活用には「理解しながら実装した記録」としてかなり使いやすいです。

### Lesson4

テーマ: 画像フォルダデータ、transform、ImageFolder、自作Dataset、TinyVGG

| ファイル | 内容 | 見せ方 |
|---|---|---|
| `Lesson4/main.py` | `torchvision.datasets.ImageFolder` とtransformの確認 | 画像分類導入 |
| `Lesson4/custom.py` | `Dataset` 継承による自作ImageFolder、TinyVGG、train/test loop、augmentation | 自作実装の強い証拠 |
| `Lesson4/practice.py` | 自作Dataset/DataLoaderの練習 | 要修正の学習ログ |
| `Lesson4/func.py` | 旧Lesson2系の補助関数 | 補助 |

実行確認:

| ファイル | 結果 | メモ |
|---|---|---|
| `Lesson4/main.py` | OK | dataset情報を出力 |
| `Lesson4/custom.py` | FAIL | `num_workers=os.cpu_count()-1` による共有メモリ/マルチプロセス周りで失敗 |
| `Lesson4/practice.py` | FAIL | `torchvision.transforms.v2` の書き方で `AttributeError: module ... has no attribute 'v2'` |
| `Lesson4/func.py` | OK | import確認 |

注意:

- `Lesson4/custom.py` は中身が非常に良いですが、実行確認ではDataLoaderのworker数が原因で失敗しました。公開前に `num_workers=0` にすると安定しやすいです。
- `Lesson4/practice.py` は現状そのまま見せるより、修正するかアーカイブ扱いがよいです。

### Lesson5

テーマ: EfficientNetB0転移学習、保存モデル、推論可視化

| ファイル | 内容 | 見せ方 |
|---|---|---|
| `Lesson5/main.py` | EfficientNetB0のfeature extractor化、pizza/steak/sushi分類、5 epoch学習 | 転移学習の成果 |
| `Lesson5/practice.py` | 同様の転移学習を20 epochで実行し、`transfer_modelv1.pth` を保存 | 成果物作成ログ |
| `Lesson5/test.py` | 保存済みモデルを読み込んで、テスト画像の予測結果を可視化 | 推論確認 |
| `Lesson5/func.py` | 補助関数 | 補助 |
| `Lesson5/model/transfer_modelv1.pth` | 保存済み転移学習モデル | 成果物候補 |

実行確認:

| ファイル | 結果 | メモ |
|---|---|---|
| `Lesson5/main.py` | TIMEOUT | 学習開始。15秒では完了せず |
| `Lesson5/practice.py` | TIMEOUT | 学習開始。15秒では完了せず |
| `Lesson5/test.py` | OK | 保存済みモデル読み込み・可視化処理まで完了 |
| `Lesson5/func.py` | OK | import確認 |

注意:

- `Lesson5/test.py` は画像表示時に正規化済みTensorをそのまま `imshow` しているため、表示範囲の警告が出ます。
- 転移学習を学んだ証拠としては使えますが、メイン成果物はLesson8の方が見せやすいです。

### Lesson6

テーマ: 実験管理、EfficientNetB0/B2比較、データ量・epoch比較、TensorBoard

| ファイル/フォルダ | 内容 | 見せ方 |
|---|---|---|
| `Lesson6/main.py` | 10%/20%データ、5/10 epoch、EffNetB0/B2の比較実験 | 実験管理の証拠 |
| `Lesson6/models/07_effnetb2_data_20_percent_10_epochs.pth` | 実験後の保存モデル候補 | 成果物候補 |
| `Lesson6/runs/` | TensorBoardログ | 評価ログ候補 |
| `Lesson6/pytorch-deep-learning/` | Learn PyTorch教材リポジトリ本体 | 公開時は分離推奨 |

実行確認:

| ファイル | 結果 | メモ |
|---|---|---|
| `Lesson6/main.py` | TIMEOUT | TensorBoard writer作成後、学習開始。15秒では完了せず |

注意:

- 実行確認により `Lesson6/runs/2026-06-28/` が生成されました。
- `Lesson6/main.py` 内で `model2` の学習時もoptimizerが `model.parameters()` を参照しているように見えるため、厳密には要確認です。
- `Lesson6/pytorch-deep-learning/` は教材のcloneに見えるため、自作ポートフォリオとしては `archive_external/` などに分けた方が安全です。

### Lesson7

テーマ: データ取得・データ準備

| ファイル | 内容 | 見せ方 |
|---|---|---|
| `Lesson7/main.py` | `download_data` でpizza/steak/sushiデータを取得・確認 | 補助的なデータ準備 |
| `Lesson7/helper_functions.py` | Learn PyTorch由来の補助関数 | 補助 |

実行確認:

| ファイル | 結果 | メモ |
|---|---|---|
| `Lesson7/main.py` | OK | 既存データがあるためdownloadをskip |
| `Lesson7/helper_functions.py` | OK | import確認 |

注意:

- 単体成果物というより、後続Lessonの準備として扱うのが自然です。

### Lesson8

テーマ: FoodVision Mini、EfficientNetB2、Gradio、Hugging Face Spaces向けデモ

| ファイル/フォルダ | 内容 | 見せ方 |
|---|---|---|
| `Lesson8/main.py` | EfficientNetB2作成、学習済みモデル読み込み、推論、Gradioアプリ化 | メイン成果物の元実装 |
| `Lesson8/models/09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth` | FoodVision Miniの学習済みモデル | メイン成果物 |
| `Lesson8/demos/my_project/app.py` | Gradio推論アプリ | 最優先で見せる |
| `Lesson8/demos/my_project/model.py` | EfficientNetB2モデル定義 | 最優先で見せる |
| `Lesson8/demos/my_project/README.md` | Hugging Face Spaces用設定 | 公開用 |
| `Lesson8/demos/foodvision_mini` | 同系統のデモ | 重複候補 |
| `Lesson8/demos/second_project` | 同系統のデモ | 重複候補 |

実行確認:

| ファイル | 結果 | メモ |
|---|---|---|
| `Lesson8/helper_functions.py` | OK | import確認 |
| `Lesson8/main.py` | FAIL | Gradioの空きポートが見つからず失敗 |
| `Lesson8/demos/*/model.py` | OK | モデル定義は実行可能 |
| `Lesson8/demos/*/app.py` | FAIL | Gradioの空きポートが見つからず失敗 |

注意:

- Gradio失敗理由は `Cannot find empty port in range: 7860-7959` です。
- `demo.launch(server_port=xxxx)` を指定すれば解消できる可能性が高いです。
- GitHubで見せるなら `my_project`、`foodvision_mini`、`second_project` のどれか1つに統一した方がよいです。

### Lesson9_recap

テーマ: 学習後の復習プロジェクト、CNN自作、EfficientNetB0転移学習、TensorBoard

| ファイル | 内容 | 見せ方 |
|---|---|---|
| `Lesson9_recap/main.py` | 復習として画像分類を再構成。自作CNNとEfficientNetB0を作成し、TensorBoard writerでログ保存 | 学習履歴として強い |
| `Lesson9_recap/func.py` | DataLoader作成、train loop、SummaryWriter作成 | 関数分割の練習として使える |
| `Lesson9_recap/runs/` | TensorBoardログ | 学習記録 |
| `Lesson9_recap/data/` | pizza/steak/sushiのtrain/testデータ | データ |

実行確認:

| ファイル | 結果 | メモ |
|---|---|---|
| `Lesson9_recap/func.py` | OK | import確認 |
| `Lesson9_recap/main.py` | TIMEOUT | 学習開始。15秒では完了せず |

注意:

- 冒頭コメントはFlowers Datasetの5クラス分類ですが、実データはpizza/steak/sushiの3クラスに見えます。公開前に説明を統一した方がよいです。
- `effnetb0.classifier` は3クラス出力になっています。
- `train_loop` は訓練のみで、test評価までは現状弱いです。

## 実行確認まとめ

### そのまま完了したもの

- `Lesson1/main.py`
- `Lesson1/mnist_digit.py`
- `Lesson1/nn.py` ※ `PYTHONPATH` 調整あり
- `Lesson1/nn2.py` ※ `PYTHONPATH` 調整あり
- `Lesson1/helper_functions.py`
- `Lesson1/test.py`
- `Lesson2/main.py`
- `Lesson2/func.py`
- `Lesson2/helper_functions.py`
- `Lesson4/main.py`
- `Lesson4/func.py`
- `Lesson5/test.py`
- `Lesson5/func.py`
- `Lesson7/main.py`
- `Lesson7/helper_functions.py`
- `Lesson8/helper_functions.py`
- `Lesson8/demos/*/model.py`
- `Lesson9_recap/func.py`

### 学習が始まるが短時間では終わらなかったもの

- `Lesson2/practice.py`
- `Lesson5/main.py`
- `Lesson5/practice.py`
- `Lesson6/main.py`
- `Lesson9_recap/main.py`

これらは「壊れている」よりも「学習スクリプトなので時間がかかる」と見るのが自然です。

### 現状失敗したもの

| ファイル | 原因 | 対応案 |
|---|---|---|
| `Lesson4/custom.py` | DataLoaderのworker数が多く、共有メモリ周りで失敗 | `num_workers=0` にする |
| `Lesson4/practice.py` | `torchvision.transforms.v2` の記述ミス | transform定義を修正 |
| `Lesson8/main.py` | Gradioの空きポートなし | `server_port` を指定 |
| `Lesson8/demos/*/app.py` | Gradioの空きポートなし | `server_port` を指定 |

## GitHubで見せる優先ファイル

### 最優先

| ファイル | 理由 |
|---|---|
| `Lesson8/demos/my_project/app.py` | Gradioで使える推論アプリ |
| `Lesson8/demos/my_project/model.py` | EfficientNetB2を使ったモデル定義 |
| `Lesson8/demos/my_project/README.md` | Hugging Face Spaces形式のREADME |
| `Lesson8/demos/my_project/requirements.txt` | デモ再現に必要 |
| `Lesson8/demos/my_project/examples/` | 推論例として使える |
| `Lesson8/demos/my_project/*.pth` | 学習済みモデル |

### 学習履歴として見せる

| ファイル | 理由 |
|---|---|
| `Lesson2/practice.py` | コメントが多く、CNNを理解しながら書いた跡がある |
| `Lesson4/custom.py` | Dataset自作、TinyVGG、augmentation、train/test loopがある |
| `Lesson5/main.py` | EfficientNetB0転移学習の導入 |
| `Lesson6/main.py` | モデル・データ量・epoch比較、TensorBoardログ |
| `Lesson9_recap/main.py` | 学習後に自力で再構成しようとした復習ログ |
| `Lesson9_recap/func.py` | 関数分割とSummaryWriterの練習 |

### 補助的に残す

| ファイル | 理由 |
|---|---|
| `Lesson1/nn.py` | 2値分類の基礎 |
| `Lesson1/nn2.py` | 多クラス分類の基礎 |
| `Lesson1/mnist_digit.py` | MNIST基礎モデル |
| `Lesson2/main.py` | FashionMNISTの評価・混同行列 |
| `Lesson5/test.py` | 保存モデルの推論可視化 |

## 公開前の整理候補

削除または `archive/` 行き候補:

- `basic.py`
- `play.py`
- `test.py`
- `media10.ipynb`
- `__pycache__/`
- `.DS_Store`
- `Lesson6/pytorch-deep-learning/`
- 重複した `Lesson8/demos/foodvision_mini`
- 重複した `Lesson8/demos/second_project`
- 古い `runs/`
- 実行確認で生成された `Lesson6/runs/2026-06-28/`

残す候補:

- `Lesson8/demos/my_project`
- `Lesson8/models`
- `Lesson2/practice.py`
- `Lesson4/custom.py`
- `Lesson5/main.py`
- `Lesson6/main.py`
- `Lesson9_recap`
- データセット取得手順または小さなexample画像

## 推奨するリポジトリ構成

既存Lessonを残しつつ、見る人向けの入口を作るなら次がよいです。

```text
README.md
LESSON_SUMMARY.md
整理提案.md
portfolio/
  foodvision-mini/
    README.md
    app.py
    model.py
    requirements.txt
    examples/
    model/
lessons/
  Lesson1/
  Lesson2/
  Lesson4/
  Lesson5/
  Lesson6/
  Lesson7/
  Lesson8/
  Lesson9_recap/
archive/
```

ただし、今すぐ大きく移動するとimportパスが壊れる可能性が高いため、最初はLessonフォルダを動かさず、ルートREADMEから見せたいファイルへ誘導する形が安全です。

## READMEに書くとよい短い説明

```md
# PyTorch Learning Portfolio

PyTorchの基礎から画像分類アプリのデプロイまでを学習した記録です。

メイン成果物は `Lesson8/demos/my_project` の FoodVision Mini です。
pizza / steak / sushi の3クラス画像分類を、EfficientNetB2の転移学習とGradioで実装しています。

学習過程として、MNIST/FashionMNIST、CNN自作、カスタムDataset、転移学習、TensorBoardによる実験管理のコードも残しています。
```

## 次にやるとよいこと

1. ルートREADMEを作る
2. `.gitignore` を作る
3. Gradioデモを1つに統一する
4. `Lesson8/demos/my_project` のREADMEを就活向けに書き直す
5. 実行方法を「ルートから」「Lesson内から」のどちらかに統一する
6. `Lesson4/custom.py` とGradioポート問題だけ最低限直す
7. TensorBoardログから最終accuracy/lossを拾う
