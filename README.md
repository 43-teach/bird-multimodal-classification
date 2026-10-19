# bird-multimodal-classification
基于 ResNet+BERT 的多模态鸟类图文分类

## 项目介绍
本项目实现图像+文本多模态鸟类分类，包含图像编码器、文本编码器以及多模态融合分类模型。
- 图像编码器：ResNet 提取视觉特征
- 文本编码器：bert-base-chinese 提取中文文本特征
- 融合方案：特征拼接融合

## 文件结构
├── datasets/                # 标注 json 文件
├── image_encoder.py         # ResNet 图像编码器
├── text_encoder.py          # BERT 文本编码器
├── 04_multimodal_fusion.py  # 融合训练与测试主程序
├── requirements.txt         # 依赖包
└── .gitignore

## 运行方式
1. 安装依赖
```bash
pip install -r requirements.txt
2. 启动训练测试

```
python 04_multimodal_fusion.py
```

## 实验说明

数据集为自制模拟数据集，样本数量较少，模型存在过拟合，测试准确率达到 100%。
