# GNSS LOS/NLOS 论文 PyTorch 复现

本目录复现论文 **“A Spatiotemporal Information-Driven Cross-Attention Model With Sparse Representation for GNSS NLOS Signal Classification”** 的系统设计和实验流程，并将原文 TensorFlow 实现改写为 PyTorch。当前 PDF 未附带原始 27 个采集点数据，因此工程提供两种入口：

- 使用真实 GNSS CSV 直接训练、评估 in-domain 和 out-domain 实验。
- 使用合成数据先跑通完整流程，验证代码、张量形状和训练闭环。

## 论文对应关系

输入特征采用论文第 IV-A 节的四类观测量：

- `cn0`: carrier-to-noise ratio, C/N0
- `elevation`: satellite elevation angle, EL
- `azimuth`: satellite azimuth angle, Az
- `pseudorange_residual`: pseudorange residual, PRE

模型实现位置：

- `src/data_np.py`: 由扁平 GNSS 观测表构造两个输入通道，使用标准库 `csv` 与 `numpy`，不依赖 pandas。
- `src/models.py`: LSTM-TFE、AAM、自注意力、cross-attention 融合、可学习稀疏激活、三层分类器。
- `src/train.py`: 训练、评估、in-domain/out-domain 划分、对比模型和消融模型。

已实现模型：

- `proposed`: AAM + LSTM-TFE + cross-attention + learnable sparse regularizer。
- `fusion`: 去掉 sparse block 的 cross-attention 融合消融。
- `concate`: 直接拼接空间/时间表征的消融。
- `mlp`: 只使用目标卫星瞬时特征的基线。
- `tbm`: Transformer/AAM 空间环境特征模型。
- `fcnn_lstm`: FCNN-LSTM 对比模型。

## 数据格式

真实数据请整理成 CSV，每行是一颗卫星在一个历元的观测：

```csv
location_id,epoch,satellite_id,label,cn0,elevation,azimuth,pseudorange_residual
L01,0,G01,1,42.1,67.2,125.0,0.3
L01,0,G02,0,30.4,18.1,201.5,6.8
```

`label=1` 表示 LOS，`label=0` 表示 NLOS；这与论文指标定义中 TP 为 LOS 一致。代码也兼容一些常见别名，如 `C/N0`、`EL`、`Az`、`PRE`、`SVID` 等。

## UrbanNav-Medium 数据集转换

如果已按当前目录结构下载 UrbanNav 官方仓库说明文件和 Medium Urban 数据：

- `data/UrbanNavDataset-master/`: UrbanNav 官方 GitHub main 分支说明、标定文件和工具。
- `data/urbannav-medium/`: `UrbanNav-HK-Medium-Urban-1` 的 GNSS、groundtruth、skymask 和 virtual sky-pointing camera video。

可直接生成本项目训练所需 CSV：

```bash
conda run -n deep_learning python scripts/make_urbannav_dataset.py \
  --root data/urbannav-medium \
  --receiver google.pixel4 \
  --out data/urbannav_medium_google_pixel4.csv
```

转换脚本会从 `.nmea` 的 GSV 语句提取 `cn0/elevation/azimuth`，从 `.obs` 的 RINEX `C1C` 伪距生成可复现的局部伪距残差特征，并用 groundtruth 最近位置匹配 skymask：卫星仰角高于对应方位角遮挡角时标为 LOS，否则标为 NLOS。

生成后即可训练主模型：

```bash
conda run -n deep_learning python src/train.py \
  --data data/urbannav_medium_google_pixel4.csv \
  --model proposed \
  --split in_domain
```

## 快速运行

安装依赖：

```bash
pip install -r requirements.txt
```

生成合成数据并跑 proposed 模型：

```bash
python scripts/make_synthetic_data.py --out data/synthetic_gnss.csv
python src/train.py --data data/synthetic_gnss.csv --model proposed --split in_domain --epochs 3
```

运行 out-domain 实验，模拟论文 “22 个地点训练、5 个未知地点测试”：

```bash
python src/train.py --data data/synthetic_gnss.csv --model proposed --split out_domain --epochs 3
```

指定未知测试地点：

```bash
python src/train.py --data data/real_gnss.csv --split out_domain --test-locations L23,L24,L25,L26,L27
```

运行消融或对比模型：

```bash
python src/train.py --data data/real_gnss.csv --model fusion
python src/train.py --data data/real_gnss.csv --model concate
python src/train.py --data data/real_gnss.csv --model tbm
python src/train.py --data data/real_gnss.csv --model fcnn_lstm
python src/train.py --data data/real_gnss.csv --model mlp
```

## 论文训练设置

默认参数按论文描述设置：

- batch size: `16`
- optimizer: Adam
- betas: `(0.9, 0.98)`
- learning rate: `0.001`
- temporal sliding window: 默认 `10`，论文中 8 或 10 表现最好
- metrics: Accuracy、Precision、Recall、F1

论文没有在可提取文本中给出所有表格超参数的精确数字，当前默认采用文中 variation 分析推荐的轻量配置：`heads=1`、`hidden_dim=64`、`ff_dim=256`、`aam_layers=1`、较高 dropout。可通过命令行覆盖。
