from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LearnableSparseActivation(nn.Module):
    """Three-stage learnable sparse regularizer from the paper.

    Parameters are constrained as w1,w2 > 0 and 0 <= b1 <= b2 through
    softplus transforms. The module implements Eq. (2), used as an
    activation block after cross-attention fusion.
    """

    def __init__(self, repeats: int = 3) -> None:
        super().__init__()
        # 稀疏激活函数的重复应用次数；论文中使用多阶段稀疏化来增强非线性表达。
        self.repeats = repeats
        # 使用 raw_* 参数保存无约束变量，再在 _params 中通过 softplus 映射到合法范围。
        self.raw_w1 = nn.Parameter(torch.tensor(0.0))
        self.raw_w2 = nn.Parameter(torch.tensor(0.0))
        self.raw_b1 = nn.Parameter(torch.tensor(-1.0))
        self.raw_delta_b = nn.Parameter(torch.tensor(0.0))

    def _params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # softplus 保证 w1、w2 为正；加 1e-6 避免数值上出现严格为 0 的斜率。softplus(x) = log(1 + exp(x))
        w1 = F.softplus(self.raw_w1) + 1e-6
        w2 = F.softplus(self.raw_w2) + 1e-6
        # b1 非负，b2 由 b1 加上一个非负增量得到，因此自然满足 0 <= b1 <= b2。
        b1 = F.softplus(self.raw_b1)
        b2 = b1 + F.softplus(self.raw_delta_b)
        return w1, w2, b1, b2

    """
        torch.where 在这里用来做按条件逐元素选择，相当于张量版的 if else。
        torch.where(condition, x, y) 返回一个张量，其中 condition 为 True 的位置取 x 的值，为 False 的位置取 y 的值。
        但注意：它不是对整个张量只判断一次，而是对张量里的每个元素分别判断。
    """
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w1, w2, b1, b2 = self._params()
        out = z
        for _ in range(self.repeats):
            # 分段线性稀疏激活：
            # 中间区间 (-b1, b1) 被压成 0，两侧区间按不同斜率线性映射。
            out = torch.where(
                out >= b2,
                w2 * (out - b2) + w1 * (b2 - b1),
                torch.where(
                    out >= b1,
                    w1 * (out - b1),
                    torch.where(
                        out > -b1,
                        torch.zeros_like(out),
                        torch.where(
                            out >= -b2,
                            w1 * (out + b1),
                            w2 * (out + b2) + w1 * (b1 - b2),
                        ),
                    ),
                ),
            )
        return out


class FeedForward(nn.Module):
    """Transformer 块中的前馈网络，将隐藏维度升维后再投影回 hidden_dim。"""

    def __init__(self, hidden_dim: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AirspaceAttentionModule(nn.Module):
    """Transformer-encoder style AAM for same-epoch satellite sequences."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        ff_dim: int = 256,
        heads: int = 1,
        layers: int = 1,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        # 将原始卫星特征映射到统一隐藏维度，作为注意力模块的输入表示。
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            # 每个块由多头自注意力、残差归一化、前馈网络和第二个残差归一化组成。
            """
                query、key、value 不要求整体形状完全一致。
                必须一致：
                batch_size 一致
                最后一维 hidden_dim 一致

                可以不一致：
                query 的序列长度 Lq
                key/value 的序列长度 Lk，但是key和 value 的序列长度必须一致，因为它们是成对出现的。
                对于自注意力来说，query、key、value 的序列长度通常相同，但在交叉注意力中，query 和 key/value 的序列长度可以不同。
            """
            self.blocks.append(
                nn.ModuleDict(
                    {
                        # attn, weights = block["mha"](query=x,key=x,value=x,key_padding_mask=spatial_mask,need_weights=False)；这里的 query、key、value 都是 x，因为是自注意力。key_padding_mask 用于告诉注意力机制哪些位置是 padding，应当忽略。need_weights=False 表示不返回注意力权重，因为我们不需要它们进行分析或可视化。
                        "mha": nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True), # 多头自注意力机制，batch_first=True 使输入输出的 batch 维在第一位。
                        "norm1": nn.LayerNorm(hidden_dim),
                        "ff": FeedForward(hidden_dim, ff_dim, dropout),
                        "norm2": nn.LayerNorm(hidden_dim),
                    }
                )
            )

    def forward(self, spatial: torch.Tensor, spatial_mask: torch.Tensor | None = None) -> torch.Tensor:
        # spatial: [batch_size, satellite_count, input_dim]
        # spatial_mask: [batch_size, satellite_count]，True 表示该位置为 padding，应被注意力忽略。
        x = self.embedding(spatial)
        for block in self.blocks:
            # 同一历元内的卫星序列做自注意力，建模卫星之间的空间/空域相关性。模型结构详情见论文 Fig.7。
            attn, _ = block["mha"](x, x, x, key_padding_mask=spatial_mask, need_weights=False) # 因为是自注意力，所以 query、key、value 都是 x。key_padding_mask 用于告诉注意力机制哪些位置是 padding，应当忽略。
            x = block["norm1"](x + attn)
            ff = block["ff"](x)
            x = block["norm2"](x + ff)
        return x # x 的形状仍然是 [batch_size, satellite_count, hidden_dim]，每个卫星都有一个对应的 hidden_dim 维特征表示。


class LSTMTemporalFeatureExtractor(nn.Module):
    """使用 LSTM 提取目标卫星的时间序列特征。"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        # PyTorch 的 LSTM 只有在层数大于 1 时才会在层间应用 dropout。
        lstm_dropout = dropout if layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, dropout=lstm_dropout, batch_first=True)

    def forward(self, temporal: torch.Tensor) -> torch.Tensor:
        # temporal: [batch_size, seq_len, input_dim]
        # batch_size: 一批样本的数量，seq_len: 时间序列长度，也就是有多少个时间步，input_dim: 每个时间步的输入特征维度。
        # hidden[-1] 是最后一层 LSTM 在最后时间步的隐藏状态，形状为 [batch_size, hidden_dim]。
        # output, (hidden, cell) = self.lstm(temporal)；其中output是所有时间步的输出，形状是： [batch_size, seq_len, hidden_dim]；hidden是所有层最后时间步的隐藏状态，形状是 [num_layers, batch_size, hidden_dim]；cell是所有层最后时间步的细胞状态，形状也是 [num_layers, batch_size, hidden_dim]。
        # 一般只需要整段时间序列的一个整体表示时，用 hidden[-1]。cell 更像“长期记忆”，hidden 更像“当前时刻对外表达出来的摘要”。
        _, (hidden, _) = self.lstm(temporal)
        return hidden[-1]


class Classifier(nn.Module):
    """二分类/回归式输出头，最终输出每个样本一个 logit。"""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # squeeze(-1) 将 [batch_size, 1] 压缩为 [batch_size]，便于后续 loss 计算。
        return self.net(x).squeeze(-1)


class SpatioTemporalCrossAttentionModel(nn.Module):
    """Proposed model: LSTM-TFE + AAM + cross-attention + sparse block."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        ff_dim: int = 256,
        heads: int = 1,
        aam_layers: int = 1,
        lstm_layers: int = 1,
        dropout: float = 0.5,
        use_cross_attention: bool = True,
        use_sparse: bool = True,
    ) -> None:
        super().__init__()
        # 时间分支：对目标卫星历史序列编码，得到 ht。
        self.temporal = LSTMTemporalFeatureExtractor(input_dim, hidden_dim, lstm_layers, dropout)
        # 空间分支：对同一历元的卫星集合编码，得到 hs。
        self.spatial = AirspaceAttentionModule(input_dim, hidden_dim, ff_dim, heads, aam_layers, dropout)
        self.use_cross_attention = use_cross_attention
        # 交叉注意力：用时间特征作为 query，到空间特征序列中检索相关信息。
        self.cross_attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True) # 这个MHA就是用来融合时间特征和空间特征的，query 是时间特征，key 和 value 都是空间特征序列。
        self.cross_norm = nn.LayerNorm(hidden_dim)
        # 非交叉注意力消融分支中，将时间特征和空间池化特征拼接后压回 hidden_dim。
        self.concat_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        # proposed 模型启用可学习稀疏激活；消融模型中可替换为恒等映射。
        self.sparse = LearnableSparseActivation(repeats=3) if use_sparse else nn.Identity()
        self.classifier = Classifier(hidden_dim, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # batch["temporal"]: [B, T, input_dim]，目标卫星时间序列。
        ht = self.temporal(batch["temporal"]) # [B, hidden_dim]，目标卫星的时间特征表示。
        # batch["spatial"]: [B, N, input_dim]，同一历元的 N 颗卫星特征。
        hs = self.spatial(batch["spatial"], batch.get("spatial_mask")) # 看看有没有mask，如果有就传进去，如果没有就传 None。AAM 模块会根据这个 mask 来忽略掉 padding 的位置，避免它们对注意力计算产生干扰。[B, N, hidden_dim]，同一历元卫星的空间特征表示序列。

        if self.use_cross_attention:
            # MultiheadAttention 期望 query 也是序列，因此给 ht 增加长度为 1 的序列维度。
            query = ht.unsqueeze(1)
            # Attention模块的输出维度跟 query 的维度一致，都是 [B, 1, hidden_dim]。这里的 fused 是融合了时间特征和空间特征的表示，后续会通过残差连接和 LayerNorm 进一步处理。
            fused, _ = self.cross_attention(query, hs, hs, key_padding_mask=batch.get("spatial_mask"), need_weights=False)
            # 残差连接 + LayerNorm 后去掉序列维度，得到融合表示 z。
            z = self.cross_norm(query + fused).squeeze(1) # [B, hidden_dim]，融合了时间特征和空间特征的表示。
        else:
            # 消融分支：不做交叉注意力，仅对空间序列做平均池化后与时间特征拼接。
            mask = batch.get("spatial_mask")
            if mask is not None:
                # mask 中 True 表示无效位置；取反后作为有效卫星的权重。
                valid = (~mask).unsqueeze(-1).float() # ~ 对布尔张量表示逻辑取反。之后的 unsqueeze(-1) 是为了在最后一维增加一个维度，使 valid 的形状变为 [B, N, 1]，这样就可以和 hs 的形状 [B, N, hidden_dim] 进行逐元素乘法了。乘积 hs * valid 会将无效位置的特征置零，而有效位置的特征保持不变。之后对这个加权后的特征求和，并除以有效卫星的数量（valid.sum(dim=1)）得到平均池化的结果。
                # clamp_min 防止某个样本全部被 mask 时除以 0。pooled = 有效卫星特征之和 / 有效卫星数量
                pooled = (hs * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            else:
                pooled = hs.mean(dim=1)
            z = self.concat_projection(torch.cat([ht, pooled], dim=-1)) # [B, hidden_dim]，拼接后投影回 hidden_dim。

        # 稀疏激活后送入分类器，输出每个样本的 logit。
        return self.classifier(self.sparse(z))

# 下面的三种模型是对比试验的基线模型，分别对应论文中的 MLP、TBM 和 FCNN-LSTM 分支。它们的设计都比较简单，主要用于验证我们提出的时空交叉注意力和可学习稀疏激活模块的有效性。
class MLPBaseline(nn.Module):
    """只使用当前目标卫星 instant 特征的 MLP 基线模型。"""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, dropout: float = 0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # batch["instant"]: [batch_size, input_dim]，当前时刻目标卫星的特征。
        return self.net(batch["instant"]).squeeze(-1)


class TransformerBasedModel(nn.Module):
    """TBM comparison: AAM/spatial Transformer plus target satellite readout."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, ff_dim: int = 256, heads: int = 1, dropout: float = 0.5) -> None:
        super().__init__()
        # 仅使用空间 Transformer/AAM 编码同一历元卫星集合。
        self.spatial = AirspaceAttentionModule(input_dim, hidden_dim, ff_dim, heads, layers=1, dropout=dropout)
        self.classifier = Classifier(hidden_dim, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor: # batch["spatial"]: [B, N, input_dim]，同一历元的 N 颗卫星特征；batch["target_index"]: [B]，目标卫星在空间序列中的位置索引。
        hs = self.spatial(batch["spatial"], batch.get("spatial_mask")) # hs 的形状是 [batch_size, satellite_count, hidden_dim]，每个卫星都有一个对应的 hidden_dim 维特征表示。我们需要从中选出目标卫星对应的特征来进行分类。
        # target_index 给出目标卫星在 spatial 序列中的位置；expand 后用于 gather。
        idx = batch["target_index"].view(-1, 1, 1).expand(-1, 1, hs.shape[-1]) # view(-1, 1, 1) 将 [B] 变为 [B, 1, 1]，expand(-1, 1, hs.shape[-1]) 将其扩展为 [B, 1, hidden_dim]，这样就可以在 gather 时指定每个样本的目标卫星位置，并且能够取出对应的 hidden_dim 维特征。
        target = hs.gather(dim=1, index=idx).squeeze(1) # 就是沿着 dim=1，也就是卫星维度 N，按 idx 指定的位置取出目标卫星的特征。取完之后形状先是：[B, 1, hidden_dim]再通过：.squeeze(1)去掉中间那个长度为 1 的维度，变成：[B, hidden_dim]
        return self.classifier(target)


class FCNNLSTMModel(nn.Module):
    """FCNN-LSTM comparison: target FCNN plus LSTM over same-epoch satellite set."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, dropout: float = 0.5) -> None:
        super().__init__()
        # 对目标卫星当前特征做前馈编码。
        self.target_fcnn = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        # 对同一历元卫星集合按序输入 LSTM，作为比较模型的空间序列编码器。
        self.spatial_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = Classifier(hidden_dim, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target = self.target_fcnn(batch["instant"]) # target 的形状是 [batch_size, hidden_dim]，是目标卫星当前特征的前馈编码表示。
        _, (hidden, _) = self.spatial_lstm(batch["spatial"]) # hidden 的形状是 [num_layers, batch_size, hidden_dim]，取最后一层的隐藏状态 hidden[-1] 作为空间特征表示，形状是 [batch_size, hidden_dim]。
        # 拼接目标当前特征与空间 LSTM 最后一层隐藏状态，再投影到分类器输入维度。
        fused = self.proj(torch.cat([target, hidden[-1]], dim=-1)) # fused 的形状是 [batch_size, hidden_dim]，融合了目标当前特征和空间序列特征的表示。
        return self.classifier(fused)


def build_model(name: str, **kwargs: object) -> nn.Module:
    """根据名称构建对应模型，供训练脚本通过配置选择不同实验分支。"""

    key = name.lower()
    # proposed: 完整模型，启用交叉注意力和可学习稀疏激活。
    if key == "proposed":
        return SpatioTemporalCrossAttentionModel(use_cross_attention=True, use_sparse=True, **kwargs)
    # fusion: 保留交叉注意力，关闭稀疏激活，用于消融稀疏模块。
    if key == "fusion":
        return SpatioTemporalCrossAttentionModel(use_cross_attention=True, use_sparse=False, **kwargs)
    # concate: 不使用交叉注意力，改为时间特征与空间池化特征拼接。
    if key == "concate":
        return SpatioTemporalCrossAttentionModel(use_cross_attention=False, use_sparse=False, **kwargs)
    # 以下为论文/实验中的对比基线模型。
    if key == "mlp":
        return MLPBaseline(input_dim=kwargs.get("input_dim", 4), hidden_dim=kwargs.get("hidden_dim", 64), dropout=kwargs.get("dropout", 0.5)) # kwargs.get("input_dim", 4) 表示如果在调用 build_model 时没有传入 input_dim 参数，就使用默认值 4。
    if key == "tbm":
        return TransformerBasedModel(
            input_dim=kwargs.get("input_dim", 4),
            hidden_dim=kwargs.get("hidden_dim", 64),
            ff_dim=kwargs.get("ff_dim", 256),
            heads=kwargs.get("heads", 1),
            dropout=kwargs.get("dropout", 0.5),
        )
    if key == "fcnn_lstm":
        return FCNNLSTMModel(input_dim=kwargs.get("input_dim", 4), hidden_dim=kwargs.get("hidden_dim", 64), dropout=kwargs.get("dropout", 0.5))
    raise ValueError(f"Unknown model: {name}")
