# PyTorch Learning Portfolio

PyTorchの基礎から画像分類アプリの公開までを学習した記録です。

## Main Project

### FoodVision Mini

`my_project/` に、現在Hugging Face向けに公開している食品画像分類アプリを置いています。

- 対象クラス: pizza / steak / sushi
- モデル: EfficientNetB2 feature extractor
- UI: Gradio
- 主なファイル:
  - `my_project/app.py`
  - `my_project/model.py`
  - `my_project/requirements.txt`
  - `my_project/09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth`

## Learning Records

- `learning_records/Lesson2/practice.py`: MNIST CNNをコメント付きで実装
- `learning_records/Lesson4/custom.py`: カスタムDataset、TinyVGG、画像分類
- `learning_records/Lesson5/main.py`: EfficientNetB0による転移学習
- `learning_records/Lesson6/main.py`: EfficientNetB0/B2、データ量、epoch比較
- `learning_records/Lesson9_recap/`: 学習後に再構成した復習プロジェクト

## Model Artifacts

`model_artifacts/` には、デモとは別に残しておきたい学習済みモデルを置いています。

## Archive

`archive/original_workspace/` には整理前のファイルを退避しています。削除はしていませんが、GitHub公開時には基本的に含めない想定です。
