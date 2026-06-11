# LOS-NLOS-main 数据划分分析

本文分析 `data/LOS-NLOS-main` 中另一套 STCA 交叉注意力 + LSTM 实现如何使用 `P2.csv` 到 `P8.csv` 这 7 个静态 GNSS 数据集，并重点说明同地点 `indomain` 和不同地点 `outdomain` 的训练、验证、测试划分。

## 结论概览

正式训练流程只生成并使用 `train/test` 两部分数据，没有实际生成 `val` 数据集。

`STCAModel.fit()` 的函数签名预留了 `X_val_spatial`、`X_val_temporal`、`y_val`，并且内部有验证集 loss/acc/F1 计算逻辑，但 `work/train_static.py`、窗口长度消融、基线消融等正式脚本都没有传入这些参数。因此当前主流程是“训练集训练，测试集最终评估”，不是 train/val/test 三划分。

## 数据预处理流程

原项目的预处理入口是 `stca/data_loading/main.py` 中的 `StaticPreprocessor.process_stca()`。整体顺序是：

```text
P2-P8 原始 CSV
-> 加载合并并添加 location
-> 过滤异常值
-> 生成 temporal/spatial 双通道样本
-> 按 indomain 或 outdomain 划分 train/test
-> 仅用 train 拟合 scaler
-> 保存 static_processed_{split_mode}.npz
```

关键代码依据：

- `loaders.py:30-38` 只发现文件名前缀在 `["P2", ..., "P8"]` 中的 CSV；`loaders.py:78-85` 合并所有 CSV，并给每行添加 `location = 文件名`。
- `filters.py:61-81` 删除 NaN，过滤 `Pr_rate_consitency == 9999.0`，过滤 `abs(Pseudorange_residual) >= 100`。
- `constants.py:14-16` 将 `LOS/NLOS_label` 映射为 `-1 -> 0`、`1 -> 1`。
- `windowers.py:83-124` 按 `(location, PRN)` 生成时间窗口；窗口标签取窗口最后一帧的标签。
- `windowers.py:151-181` 按 `(GPS_Time(s), location)` 取同一时刻同一地点所有卫星，padding/truncate 到 `max_satellites`。
- `main.py:164-178` 在划分之后，只用训练集 temporal/spatial 拟合统一 scaler，再变换训练集和测试集。

## 同地点划分：indomain

同地点划分的意思是：每个测站 `P2` 到 `P8` 内部都随机拆成训练集和测试集，训练和测试都包含所有地点。

实现位置：

- `splitters.py:36-109`
- `main.py:98-130`

具体逻辑：

```text
for 每个 location:
    找出该 location 的所有窗口样本
    设置 np.random.seed(random_seed)
    对该 location 内样本索引随机打乱
    前 int(n_samples * test_size) 个作为 test
    剩余作为 train
最后把 7 个地点的 train 拼接，7 个地点的 test 拼接
```

默认比例要分情况看：

| 入口 | test_size 来源 | 实际含义 |
|---|---:|---|
| `python -m data_loading.main --split-mode indomain` | `DEFAULT_TEST_SIZE = 0.3` | 每地点约 70% train / 30% test |
| `python -m work.train_static` 且 `.npz` 不存在、由训练脚本触发预处理 | `config["test_size"] = 0.2` | 每地点约 80% train / 20% test |

这里有一个实现不一致：README 和 `data_loading/constants.py` 写的是 70/30，但 `work/train_static.py:477` 把 `test_size` 设置成了 `0.2`。如果已有 `static_processed_indomain.npz`，训练脚本会直接加载旧 npz，比例取决于当初生成该 npz 的入口。

注意：该实现是在窗口化之后按样本随机拆分，不是按 epoch、PRN 轨迹或连续时间段拆分。因此同一地点、相邻时间窗口可能同时出现在 train 和 test 中，这更像同分布随机测试，不是严格时间外推测试。

## 不同地点划分：outdomain

不同地点划分的意思是：训练集和测试集使用完全不同的测站，测试模型跨地点泛化能力。

实现位置：

- `constants.py:38-41`
- `splitters.py:111-173`
- `main.py:131-136`

固定地点配置为：

```python
OUTDOMAIN_TRAIN_LOCATIONS = ["P2", "P3", "P4", "P5", "P8"]
OUTDOMAIN_TEST_LOCATIONS = ["P6", "P7"]
```

具体逻辑：

```text
train_mask = locations in ["P2", "P3", "P4", "P5", "P8"]
test_mask  = locations in ["P6", "P7"]
X_train/y_train = train_mask 对应样本
X_test/y_test   = test_mask 对应样本
```

这个模式不受 `test_size` 影响，也不随机选地点；P6、P7 始终作为测试地点。

消融脚本多数硬编码使用 outdomain。例如 `work/ablation_window_size.py` 的 `CONFIG["split_mode"] = "outdomain"`，基线和超参数消融也采用类似设置。

## 验证集 val 的情况

正式预处理结果只保存：

```text
X_train_temporal
X_test_temporal
X_train_spatial
X_test_spatial
y_train
y_test
```

对应代码在 `main.py:195-209` 和 `main.py:215-228`。没有 `X_val_*` 或 `y_val`。

模型训练函数 `STCAModel.fit()` 支持验证集参数：

```python
X_val_spatial=None
X_val_temporal=None
y_val=None
```

并且 `stca_model.py:565-625` 内部会在这些参数存在时计算 `val_loss`、`val_acc`、`val_f1`。但正式训练脚本 `train_static.py:527-535` 调用 `model.fit()` 时只传：

```python
model.fit(
    X_train_spatial,
    y_train,
    ...,
    X_train_temporal=X_train_temporal,
)
```

没有传任何验证集。因此 `history["val_loss"]` 等列表会保持为空，训练曲线也只画训练集 loss/acc/F1。

`data_loading/test.ipynb` 中出现过 `X_train_t, X_val_t, X_test_t, ...` 的调试代码，但当前 `splitters.py` 的 `split_outdomain()` 实际只返回 train/test 六个数组，不返回 val。这个 notebook 更像早期或临时实验痕迹，不能代表当前正式流程。

## 测试集使用方式

正式训练结束后，`work/train_static.py` 会用最终模型在 test set 上评估：

- `train_static.py:525-535`：只用训练集调用 `model.fit()`。
- `train_static.py:551-560`：构造 test loader 并调用 `evaluate_model()`。
- `train_static.py:328-356`：计算 accuracy、precision、recall、F1、ROC AUC、PR AUC。

模型没有根据验证集挑选 best checkpoint；保存的是训练结束后的最终模型 `final_model_{split_mode}.pth`。

## 与当前复现代码的建议对齐

如果我们要尽量复现对方正式实验设置：

1. 标签保持 `-1 -> 0`、`1 -> 1`，因为对方模型输出 sigmoid 概率并使用 BCE。
2. 使用 4 个特征：`C/N0`、`Elevation`、`Azimuth`、`Pseudorange_residual`。
3. 使用窗口长度 `22`，最大卫星数 `20`，对应 `constants.py:26-28`。
4. outdomain 固定为 `P2,P3,P4,P5,P8` 训练，`P6,P7` 测试。
5. indomain 需要明确选择比例：按 README/预处理默认应是每地点 70/30；按 `train_static.py` 自动预处理则会变成 80/20。
6. 若想和对方正式脚本一致，不需要 validation set；若想更规范地调参，可以在当前复现中额外从 train 内划出 val，但这已经不是原项目主流程。

