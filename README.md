# 紡織品製程分類系統

基於 Swin Transformer、CLIP 和 LBP 紋理分析的多模態深度學習紡織品製造製程分類系統。

## 環境需求

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 模型訓練

設定資料目錄路徑：
```bash
export TEXTILE_DATA_DIR=/path/to/your/fabric/data
```

執行訓練：
```bash
python train.py
```

訓練腳本需要以下資料結構：
```
data/
├── data/
│   ├── train.csv      # 訓練資料
│   ├── val.csv        # 驗證資料
│   └── test.csv       # 測試資料
└── new_augmented/     # 影像資料夾
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

CSV 檔案格式：
```csv
image,Ingredients,GT
fabric_001.jpg,"羊毛",製程工序1
fabric_002.jpg,"絲綢",製程工序2
```

### 2. 模型推理

#### 網頁介面
```bash
python inference.py --model_path best_textile_model.pt
```

在瀏覽器開啟 `http://localhost:7860` 使用網頁介面

#### 命令列介面
```bash
python inference.py --interface cli --model_path best_textile_model.pt --image fabric.jpg --ingredients "棉花, 聚酯纖維"
```

#### 程式化調用
```python
from inference import TextilePredictor

predictor = TextilePredictor('best_textile_model.pt')
results = predictor.predict(image, ingredients_text)

for class_name, confidence in results:
    print(f"{class_name}: {confidence:.2%}")
```

### 3. 配置選項

#### 訓練配置
編輯 `train.py` 中的配置字典：
```python
config = {
    'data_directory': './data',
    'batch_size': 16,
    'num_epochs': 30,
    'learning_rate': 1e-4,
    'use_texture': True,
}
```

#### 推理選項
```bash
# 基本用法
python inference.py

# 自定義模型路徑
python inference.py --model_path /path/to/model.pt

### 4. 輸出檔案

訓練完成後，將生成以下檔案：
- `best_textile_model.pt` - 最佳性能模型
- `textile_model_final.pt` - 最終模型狀態
- `confusion_matrix_normalized.png` - 歸一化混淆矩陣
- `confusion_matrix_raw.png` - 原始混淆矩陣
- `classification_report.txt` - 詳細分類指標
- `training_history.png` - 訓練損失和準確率曲線

### 5. 環境變數

```bash
# 資料目錄路徑
export TEXTILE_DATA_DIR=/path/to/data

# 可選：推理模型路徑
export TEXTILE_MODEL_PATH=/path/to/model.pt
```
