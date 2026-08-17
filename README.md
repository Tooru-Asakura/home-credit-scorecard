# Home Credit 信贷申请评分卡

> **Credit Application Scorecard (A-Card)** — 基于 Kaggle Home Credit Default Risk 数据的端到端评分卡构建，输出标准信用评分（PDO=20，基准分 600）

信贷风控的核心交付物：把客户违约风险转换为**可解释、可审计的信用分数**。本项目覆盖评分卡构建全流程——多表特征工程、三层特征筛选、WoE 编码、逻辑回归建模、评分卡刻度转换，最终产出带分箱明细的标准评分卡。

---

## 核心指标（5 折交叉验证 OOF）

| 指标 | 值 | 说明 |
|------|-----|------|
| AUC | **0.7447** | 区分度：随机 0.5，完美 1.0 |
| KS | **0.3657** | 风控行业 KS>0.3 即区分度良好 |
| Gini | **0.4893** | = 2×AUC - 1，与 AUC 换算一致 |

分箱质量示例（EXT_SOURCE_MEAN）：坏账率从 22.9%（低分箱）单调降至 4.2%（高分箱）——评分卡单调性验证通过。

---

## 流水线架构

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1  数据加载：7 张表（application/bureau/previous/       │
│         pos/credit_card/installments/bureau_balance）        │
├─────────────────────────────────────────────────────────────┤
│ Step 2  特征工程：跨表聚合（均值/计数/占比/交叉特征）          │
├─────────────────────────────────────────────────────────────┤
│ Step 3  LGB 5 折 CV → 特征重要性排序                         │
├─────────────────────────────────────────────────────────────┤
│ Step 4  三层特征筛选                                         │
│   Layer1: LGB 重要性 Top-N                                  │
│   Layer2: 相关系数 > 0.85 去共线（保留 IV 高者）              │
│   Layer3: EXT_SOURCE 业务规则过滤                            │
├─────────────────────────────────────────────────────────────┤
│ Step 5  WoE 编码（分箱 → 证据权重）→ IV 汇总                  │
├─────────────────────────────────────────────────────────────┤
│ Step 6  Logistic Regression 5 折建模                        │
├─────────────────────────────────────────────────────────────┤
│ Step 7  评分卡转换（PDO=20, 基准分 600）+ 分布校验             │
└─────────────────────────────────────────────────────────────┘
```

## 方法论亮点（面试可讲）

1. **三层特征筛选**：不是单层拍脑袋选特征——LGB 重要性粗筛 → 相关系数+IV 共线过滤 → 业务规则精筛，每一步都有明确目的
2. **WoE 编码**：把非线性关系线性化（LR 前提）、天然处理缺失值（单独分箱）、分箱粒度带 bad_rate 单调性校验
3. **评分卡刻度设计**：`Score = Offset + Factor × ln(odds)`，PDO=20（odds 翻倍加 20 分）是行业标准刻度
4. **评估用 KS/Gini 而非准确率**：信贷违约率 ~8%，准确率毫无意义；KS/Gini 衡量区分度才是风控语言
5. **可解释性**：`case_study_deep.py` 输出典型案例解剖 + 错误案例分析——每个客户的分数可以拆解到"哪个特征、哪个分箱、加了多少分"

## 目录结构

```
├── main.py                    # 全流程入口（python main.py）
├── case_study_deep.py         # 典型案例解剖 + 错误分析
├── src/
│   ├── config.py              # 路径/参数配置
│   ├── data_loader.py         # 多表数据加载
│   ├── feature_engineering.py # 跨表聚合特征
│   ├── feature_selection.py   # 三层特征筛选
│   ├── woe_encoding.py        # WoE 编码 + IV
│   ├── model_lr.py            # LR 5 折建模
│   └── scorecard.py           # 评分卡刻度转换 + 分布校验
├── notebooks/01_EDA.ipynb     # 探索性数据分析
├── docs/API_REFERENCE.md      # 模块 API 文档
└── outputs/
    ├── models/                # 5 折 LGB + LR 模型
    ├── features/              # IV 汇总 / WoE 映射 / 特征筛选结果
    └── scorecard/             # 标准评分卡 + 分数分布
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据：从 Kaggle 下载 Home Credit Default Risk 数据集
#    （application_train.csv 等 7 张表 → data/raw/）

# 3. 全流程运行（首次约 30-60 分钟，含 LGB CV + LR CV）
python main.py

# 4. 案例解剖（可解释性演示）
python case_study_deep.py
```

## 数据说明

- 数据集：[Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)（约 30 万客户，违约率 ~8%）
- 原始数据 3.2GB 不入库（`data/` 已在 .gitignore），需自行下载
- 所有中间产物（WoE 映射、IV 汇总、模型、评分卡）均已提交，可直接查看

## 文档

| 文档 | 内容 |
|------|------|
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | 模块 API 参考 + 调用链路 |
| [notebooks/01_EDA.ipynb](notebooks/01_EDA.ipynb) | 探索性数据分析 |
