# 人脸识别系统使用说明

## 项目概述

本项目是一个基于深度学习的人脸识别系统，可以对人脸进行检测、裁剪和识别。系统包含交互式界面，支持上传图片、选择人脸区域进行识别，并能够标记陌生人。整个系统使用Python编写，采用模块化设计，便于维护和扩展。

## 系统要求

### 环境依赖

```
# 核心依赖
numpy>=1.19.5
opencv-python>=4.5.1
tensorflow>=2.5.0
Pillow>=8.2.0
matplotlib>=3.4.2
pathlib>=1.0.1

# 用于处理中文字符路径
chardet>=4.0.0

# 图形界面
tk>=0.1.0
```

## 项目结构

```
facial_recognition/
├── data_processing/         # 数据处理模块
│   ├── __init__.py
│   ├── face_detection.py    # 人脸检测和裁剪
│   └── data_loader.py       # 数据集处理（支持中文字符）
├── model/                   # 模型模块
│   ├── __init__.py
│   ├── face_model.py        # 深度学习模型架构
│   └── trainer.py           # 训练流程
├── interface/               # 界面模块
│   ├── __init__.py
│   └── gui.py               # 交互式图形界面
├── utils/                   # 工具模块
│   ├── __init__.py
│   └── helpers.py           # 辅助函数
├── main.py                  # 主程序入口
└── requirements.txt         # 环境依赖说明
```

## 使用说明

### 1. 准备数据集

- 数据集应包含50个子文件夹，每个文件夹代表一个人，文件夹名为该人的姓名（支持中文）
- 每个子文件夹应包含30张该人的图片
- 系统将使用每个人的前24张图片进行训练，剩余图片用于测试

### 2. 处理数据集

```bash
python main.py --mode process --data_dir "你的数据集路径" --processed_dir "processed_faces"
```

此命令会检测所有图片中的人脸并裁剪出来，用于后续训练。

### 3. 训练模型

```bash
python main.py --mode train --data_dir "你的数据集路径" --processed_dir "processed_faces" --output_dir "output"
```

默认使用CPU训练。如需使用GPU训练，请添加`--use_gpu`参数。

### 4. 运行图形界面

```bash
python main.py --mode gui --processed_dir "processed_faces" --output_dir "output"
```

### 5. 一键完成所有步骤

```bash
python main.py --mode all --data_dir "你的数据集路径" --processed_dir "processed_faces" --output_dir "output"
```

## 图形界面使用说明

1. **上传图片**：点击"Upload Image"按钮选择要识别的图片
2. **选择人脸**：在左侧图片中点击检测到的人脸区域
3. **识别人脸**：点击"Recognize Face"按钮进行识别
4. **标记陌生人**：如果上传的图片中的人不在数据集中，可以点击"Mark as Stranger"按钮将其标记为陌生人

## 注意事项

1. **数据集路径**：请将命令中的`你的数据集路径`替换为实际的数据集路径
2. **GPU训练**：系统默认使用CPU训练，如需使用GPU，请在训练时添加`--use_gpu`参数
3. **中文支持**：系统已设计为可以处理中文字符的文件路径和文件夹名
4. **训练/测试分割**：系统会使用每个人文件夹中的前24张图片进行训练，剩余图片用于测试
5. **交互界面**：图形界面允许上传图片、选择人脸、识别人脸，以及将未识别的人脸标记为"陌生人"

## 参数说明

- `--data_dir`：原始数据集目录
- `--processed_dir`：处理后的人脸数据保存目录
- `--output_dir`：输出目录（模型、图表等）
- `--batch_size`：训练批次大小
- `--epochs`：训练轮数
- `--use_gpu`：是否使用GPU训练
- `--mode`：运行模式（process/train/gui/all）

## 系统特点

1. **模块化设计**：系统采用模块化设计，便于维护和扩展
2. **中文支持**：完全支持中文路径和文件名
3. **灵活训练**：支持CPU和GPU训练
4. **数据增强**：训练过程中使用数据增强提高模型鲁棒性
5. **交互式界面**：提供友好的图形用户界面
6. **自动人脸检测**：自动检测和裁剪人脸区域

        