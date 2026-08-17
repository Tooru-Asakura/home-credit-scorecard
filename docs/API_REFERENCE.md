# Home Credit 评分卡项目 —— API 参考文档

> 版本：v1.0 | 更新日期：2026-03-02 | 语言：Python 3.10

---

## 目录

1. [项目概览](#1-项目概览)
2. [模块文档](#2-模块文档)
   - [src/config.py](#21-srcconfigpy)
   - [src/data_loader.py](#22-srcdata_loaderpy)
   - [src/preprocessing.py](#23-srcpreprocessingpy)
   - [src/feature_engineering.py](#24-srcfeature_engineeringpy)
   - [src/feature_selection.py](#25-srcfeature_selectionpy)
   - [src/woe_encoding.py](#26-srcwoe_encodingpy)
   - [src/model_lr.py](#27-srcmodel_lrpy)
   - [src/scorecard.py](#28-srcscorecardpy)
   - [main.py](#29-mainpy)
3. [完整调用链路](#3-完整调用链路)
4. [输入输出文件说明](#4-输入输出文件说明)

---

## 1. 项目概览

**项目目标**：基于 Kaggle Home Credit Default Risk 数据集，构建一套端到端的信贷评分卡系统，输出标准评分卡（PDO=20，基准分=600），用于贷前信用风险决策。

### 1.1 技术路线流程图

```mermaid
flowchart LR
    A[原始CSV数据\n7张表] --> B[数据加载\ndata_loader]
    B --> C[数据清洗\npreprocessing]
    C --> D[特征工程\nfeature_engineering\n214个特征]
    D --> E[LGB特征筛选\nfeature_selection\nTop-40特征]
    E --> F[WoE编码\nwoe_encoding\nIV筛选→28特征]
    F --> G[共线性剔除\nmodel_lr\n23特征]
    G --> H[LR建模\n5折CV\nKS=0.37]
    H --> I[评分卡转换\nscorecard\n评分范围693~763]
```

### 1.2 模块依赖关系图

```
main.py
├── src/data_loader.py          ← 依赖: config
├── src/feature_engineering.py  ← 依赖: preprocessing, config
│   └── src/preprocessing.py   ← 无内部依赖
├── src/feature_selection.py    ← 依赖: config
├── src/woe_encoding.py         ← 依赖: config
├── src/model_lr.py             ← 依赖: config
└── src/scorecard.py            ← 依赖: config
    └── src/config.py           ← 无内部依赖（根节点）
```

### 1.3 关键性能指标（训练集结果）

| 指标 | LGB（特征筛选） | LR（评分卡） |
|------|--------------|------------|
| OOF AUC | 0.7881 | 0.7456 |
| OOF KS | — | 0.3663 ✓ |
| Gini | — | 0.4912 |
| 评分范围 | — | 693 ~ 763 |
| 好坏均分差 | — | 12.7 分 |

---

## 2. 模块文档

---

### 2.1 `src/config.py`

**模块职责**：全局配置中心，集中管理所有路径、文件名、模型超参数和评分卡参数，避免硬编码散落各处。

**依赖模块**：无（项目根节点，仅依赖标准库 `os`）

**对外提供的配置项**：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `BASE_DIR` | str | 自动推断 | 项目根目录绝对路径 |
| `DATA_RAW` | str | `data/raw/` | 原始 CSV 存放目录 |
| `DATA_PROCESSED` | str | `data/processed/` | 中间产物目录（特征缓存）|
| `OUTPUT_DIR` | str | `outputs/` | 所有输出根目录 |
| `MODEL_DIR` | str | `outputs/models/` | 模型文件目录 |
| `FEATURE_DIR` | str | `outputs/features/` | 特征重要性/WoE 映射目录 |
| `SCORECARD_DIR` | str | `outputs/scorecard/` | 评分卡输出目录 |
| `FILES` | dict | — | 7 张原始 CSV 的文件名映射 |
| `TARGET` | str | `'TARGET'` | 目标变量列名（1=违约，0=正常）|
| `ID_COL` | str | `'SK_ID_CURR'` | 申请人唯一标识列名 |
| `LGB_PARAMS` | dict | 见下表 | LightGBM 超参数 |
| `LGB_CV_FOLDS` | int | `5` | 交叉验证折数 |
| `LGB_EARLY_STOP` | int | `150` | 早停窗口轮数 |
| `IMPORTANCE_THRESHOLD` | int | `10` | LGB 重要性最低阈值 |
| `TARGET_FEATURE_NUM` | int | `40` | 特征筛选目标数量 |
| `CORRELATION_THRESHOLD` | float | `0.85` | 共线性剔除相关系数阈值 |
| `PDO` | int | `20` | Points to Double Odds |
| `BASE_SCORE` | int | `600` | 基准分 |
| `BASE_ODDS` | int | `50` | 基准好坏比（好人数/坏人数）|

#### LGB_PARAMS 详细说明

| 参数 | 值 | 业务原因 |
|------|----|---------|
| `n_estimators` | 3000 | 配合早停使用，不会真的跑满；留足迭代空间 |
| `learning_rate` | 0.05 | 特征筛选阶段合理默认值，不追求极致 AUC |
| `num_leaves` | 31 | 控制单棵树复杂度，防噪声特征 importance 虚高 |
| `max_depth` | 6 | 与 `num_leaves` 双重约束树深度 |
| `min_child_samples` | 100 | 307k 数据集每叶最少样本数，原值 20 过小会产生噪声叶节点 |
| `min_split_gain` | 0.01 | 过滤 AUC 提升极微小的无效分裂 |
| `feature_fraction` | 0.7 | 列采样；更低采样减少共线性特征"轮流上榜" |
| `bagging_fraction` | 0.8 | 行采样，防过拟合 |
| `reg_alpha` | 0.5 | L1 正则，让弱特征权重趋向 0 |
| `reg_lambda` | 1.0 | L2 正则，抑制对局部噪声的过拟合 |
| **`scale_pos_weight`** | **11** | **正负样本比 ≈ 1:11（违约率 8.1%），0.919/0.081≈11；不加此参数模型偏向多数类** |

---

### 2.2 `src/data_loader.py`

**模块职责**：负责从 `data/raw/` 目录读取所有原始 CSV 文件，返回统一的字典结构。

**依赖模块**：`src/config`（`DATA_RAW`, `FILES`）

**对外提供的函数**：

- `load_data(name)`
- `load_all()`

---

#### `load_data(name: str) -> pd.DataFrame`

**功能描述**：按名称读取单张原始 CSV 表。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `name` | str | `FILES` 字典中的 key，如 `'application'`、`'bureau'` |

**返回值**：`pd.DataFrame`，读取到的原始表

**异常**：若文件不存在抛出 `FileNotFoundError`

**调用示例**：
```python
from src.data_loader import load_data
app = load_data('application')   # 读取 application_train.csv
```

---

#### `load_all() -> dict`

**功能描述**：一次性读取所有 7 张原始表，打印各表的 shape 并返回字典。

**参数说明**：无

**返回值**：`dict`，格式为 `{表名: DataFrame}`

```python
{
    'application'   : DataFrame(307511, 122),
    'bureau'        : DataFrame(1716428, 17),
    'bureau_balance': DataFrame(27299925, 3),
    'previous'      : DataFrame(1670214, 37),
    'pos'           : DataFrame(10001358, 8),
    'credit_card'   : DataFrame(3840312, 23),
    'installments'  : DataFrame(13605401, 8),
}
```

**业务逻辑说明**：数据以申请人（`SK_ID_CURR`）为主键，`bureau`/`previous`/`pos` 等子表按申请人做 left join 聚合后才能与主表合并。

**调用示例**：
```python
from src.data_loader import load_all
data = load_all()
print(data['bureau'].shape)   # (1716428, 17)
```

---

### 2.3 `src/preprocessing.py`

**模块职责**：主表数据清洗与基础编码，修复已知异常值，为特征工程提供干净的输入。

**依赖模块**：无内部依赖（仅 numpy/pandas/sklearn）

**对外提供的函数**：

- `clean_application(df)`
- `encode_categoricals(df)`
- `fill_missing(df)`

---

#### `clean_application(df: pd.DataFrame) -> pd.DataFrame`

**功能描述**：对主表（`application_train`）执行异常值修复和基础特征衍生。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `df` | pd.DataFrame | 原始 application_train 表 |

**返回值**：`pd.DataFrame`，清洗后的主表（保留所有原始列，新增时间衍生列）

**业务逻辑说明**：

| 处理项 | 原始问题 | 处理方式 |
|--------|---------|---------|
| `DAYS_EMPLOYED=365243` | 未就业占位符，不是真实天数 | 替换为 `NaN` |
| `AGE_YEARS` | 原始 `DAYS_BIRTH` 为负数（天） | 取 `-DAYS_BIRTH/365`，转为正数年龄 |
| `CODE_GENDER='XNA'` | 仅 4 条，无效性别 | 替换为 `NaN` |
| 金额字段负值 | 数据录入错误 | `clip(lower=0)` 截断为 0 |

**调用示例**：
```python
from src.preprocessing import clean_application
app_clean = clean_application(data['application'])
```

---

#### `encode_categoricals(df: pd.DataFrame) -> pd.DataFrame`

**功能描述**：对所有 `object` 类型列进行 `LabelEncoder` 整数编码。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `df` | pd.DataFrame | 含类别变量的 DataFrame |

**返回值**：`pd.DataFrame`，类别列已替换为整数

**业务逻辑说明**：LightGBM 可直接处理整数编码的类别变量；后续 LR 阶段会用 WoE 值替换，此处只做最基础的兼容性处理。缺失值先填 `'missing'` 再编码，避免 `LabelEncoder` 报错。

**调用示例**：
```python
from src.preprocessing import encode_categoricals
df_encoded = encode_categoricals(app_clean)
```

---

#### `fill_missing(df: pd.DataFrame) -> pd.DataFrame`

**功能描述**：按字段语义分策略填充缺失值。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `df` | pd.DataFrame | 待填充的 DataFrame |

**返回值**：`pd.DataFrame`，缺失值已填充

**业务逻辑说明**：

| 字段类型 | 判断规则 | 填充值 | 原因 |
|---------|---------|--------|------|
| 计数/求和类特征 | 列名含 `COUNT`/`SUM`/`_NUM` | `0` | 无子表记录 = 0 次，语义正确 |
| 其余数值特征 | 剩余数值列 | `-999` | 缺失标记；对树模型友好（树会自动学到"缺失"这个分割点）|

**调用示例**：
```python
from src.preprocessing import fill_missing
df_filled = fill_missing(df_encoded)
```

---

### 2.4 `src/feature_engineering.py`

**模块职责**：从 7 张原始表中提取和衍生全量特征，最终输出一张宽表（307511 × 214），供 LGB 特征筛选使用。

**依赖模块**：`src/preprocessing`（`clean_application`, `encode_categoricals`, `fill_missing`）

**对外提供的函数**：

| 函数名 | 作用 |
|--------|------|
| `feat_application(df)` | 主表衍生特征（负债比、就业稳定性、EXT 交叉等）|
| `feat_bureau(bureau, bureau_bal)` | 外部征信两级聚合特征 |
| `feat_installments(inst)` | 分期还款行为特征（DPD/DBD/6M窗口/趋势）|
| `feat_credit_card(cc)` | 信用卡使用行为特征（利用率/还款比例）|
| `feat_pos(pos)` | POS/现金贷月度还款状态特征 |
| `feat_previous(prev)` | HC 历史申请记录特征 |
| `feat_cross(df)` | 跨表交叉特征（多重风险叠加信号）|
| `build_features(data)` | 主入口：合并所有子表特征，返回完整宽表 |

---

#### `feat_application(df: pd.DataFrame) -> pd.DataFrame`

**功能描述**：在主表上衍生负债压力、就业稳定性、材料完整度、社交圈风险、EXT_SOURCE 交叉等特征。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `df` | pd.DataFrame | 经过 `clean_application` 处理后的主表（已含 `AGE_YEARS`、`DAYS_EMPLOYED_YEARS`）|

**返回值**：`pd.DataFrame`，追加衍生列后的主表

**衍生特征一览**：

| 特征名 | 计算公式 | 金融含义 |
|--------|---------|---------|
| `CREDIT_INCOME_RATIO` | `AMT_CREDIT / (AMT_INCOME_TOTAL+1)` | DTI（债务收入比），值越大还款压力越大 |
| `ANNUITY_INCOME_RATIO` | `AMT_ANNUITY / (AMT_INCOME_TOTAL+1)` | 月还款收入比，反映月现金流压力 |
| `CREDIT_TERM` | `AMT_CREDIT / (AMT_ANNUITY+1)` | 贷款期限估算（年），期限越长信用暴露越高 |
| `CREDIT_GOODS_RATIO` | `AMT_CREDIT / (AMT_GOODS_PRICE+1)` | 套现信号，`>>1` 说明贷款远超商品价值 |
| `INCOME_PER_PERSON` | `AMT_INCOME_TOTAL / (CNT_FAM_MEMBERS+1)` | 家庭人均收入，比总收入更能反映还款余力 |
| `EMPLOYED_TO_AGE_RATIO` | `DAYS_EMPLOYED_YEARS / (AGE_YEARS+1)` | 就业年限占年龄比，高值代表职业稳定 |
| `DOCUMENT_COUNT` | `FLAG_DOCUMENT_*` 求和 | 提交材料数量，少→信息不透明→风险高 |
| `CONTACT_COUNT` | 联系方式标志求和 | 可触达性，高→催收容易→风险偏低 |
| `SOCIAL_CIRCLE_DEF_RATE_30` | `DEF_30 / (OBS_30+1)` | 30天社交圈违约率，信用风险具传染性 |
| `SOCIAL_CIRCLE_DEF_RATE_60` | `DEF_60 / (OBS_60+1)` | 60天社交圈违约率，更长观察窗口 |
| `EXT_SOURCE_MEAN` | 三源均值（忽略缺失）| 综合外部征信评分，是数据集最强预测特征 |
| `EXT_SOURCE_STD` | 三源标准差 | 评分离散度越大→不同机构评价分歧→信用画像不稳定 |
| `EXT_SOURCE_PROD` | 三源乘积 | 同时惩罚所有维度偏低的情况 |
| `EXT_SOURCE_1_2` | `EXT_SOURCE_1 × EXT_SOURCE_2` | 两机构评分协同效应 |
| `EXT_SOURCE_2_3` | `EXT_SOURCE_2 × EXT_SOURCE_3` | 最强单特征与3的交叉 |
| `EXT_SOURCE_2_AGE` | `EXT_SOURCE_2 × AGE_YEARS` | 外部评分×年龄，放大"老+好"组合信号 |

**调用示例**：
```python
from src.feature_engineering import feat_application
app = feat_application(app_clean)  # 输入已经过 clean_application 处理
```

---

#### `feat_bureau(bureau: pd.DataFrame, bureau_bal: pd.DataFrame) -> pd.DataFrame`

**功能描述**：对外部征信数据进行两级聚合，最终输出以 `SK_ID_CURR`（申请人）为粒度的聚合特征表。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `bureau` | pd.DataFrame | 外部征信单笔借据表（1,716,428 行 × 17 列）|
| `bureau_bal` | pd.DataFrame | 外部征信月度还款快照表（27,299,925 行 × 3 列）|

**返回值**：`pd.DataFrame`，以 `SK_ID_CURR` 为主键的聚合特征表（约 305k 行）

**业务逻辑说明——两级聚合层次**：

```
bureau_balance（每笔贷款的月度状态）
    ↓ Level-1 聚合：按 SK_ID_BUREAU 汇总
bureau（每笔贷款的基本信息）
    ↓ Level-2 聚合：按 SK_ID_CURR 汇总
application（申请人级别）
```

**Level-1 中间特征（`SK_ID_BUREAU` 粒度）**：

| 特征 | 含义 |
|------|------|
| `BB_MONTH_COUNT` | 该笔贷款存续月数 |
| `BB_DPD_COUNT` | 历史逾期月数（`STATUS >= 1`）|
| `BB_DPD_RATE` | 逾期月比率 = 逾期月数 / 总月数 |
| `BB_STATUS_MAX` | 历史最严重逾期等级（1~5）|
| `BB_STATUS_MEAN` | 平均逾期等级 |

> `STATUS` 映射：`C/X/0 → 0`（无逾期），`1~5 → 1~5`（逾期月数级别）

**Level-2 输出特征（`SK_ID_CURR` 粒度）**：

| 特征名 | 金融含义 |
|--------|---------|
| `BUREAU_COUNT` | 历史外部借贷笔数 |
| `BUREAU_ACTIVE_COUNT/RATE` | 当前仍激活贷款数/比例 |
| `BUREAU_CREDIT_SUM_MEAN/MAX` | 单笔外部授信均值/最大值 |
| `BUREAU_DEBT_SUM/MEAN` | 外部未偿债务总量/均值 |
| `BUREAU_OVERDUE_MEAN/MAX` | 逾期金额均值/最大值 |
| `BUREAU_DAYS_CREDIT_MEAN/MAX` | 借贷记录年龄（天）|
| `BUREAU_UTIL_MEAN/MAX` | 外部贷款使用率（已用/批准）|
| `BUREAU_PROLONG_SUM` | 历史展期次数（软性违约信号）|
| `BUREAU_BB_DPD_RATE_MEAN` | 月度逾期率均值（跨级聚合）|
| `BUREAU_BB_STATUS_MAX` | 历史最严重逾期等级（跨级聚合）|

**调用示例**：
```python
from src.feature_engineering import feat_bureau
bureau_feat = feat_bureau(data['bureau'], data['bureau_balance'])
# 返回 shape ≈ (305811, 18)，以 SK_ID_CURR 为主键
```

---

#### `feat_installments(inst: pd.DataFrame) -> pd.DataFrame`

**功能描述**：从分期还款流水表中提取还款纪律特征，包括全量历史聚合、近 6 个月窗口聚合和趋势特征。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `inst` | pd.DataFrame | `installments_payments.csv`，13,605,401 行，每行一次应还/实还事件 |

**返回值**：`pd.DataFrame`，以 `SK_ID_CURR` 为主键的聚合特征表

**中间衍生变量**：

| 变量 | 公式 | 含义 |
|------|------|------|
| `DPD` | `(实还日 - 应还日).clip(0)` | 逾期天数（正数），提前还款截断为 0 |
| `DBD` | `(应还日 - 实还日).clip(0)` | 提前还款天数（正数），逾期截断为 0 |
| `PAYMENT_RATIO` | `AMT_PAYMENT / (AMT_INSTALMENT+1)` | 还款比例：1=全额，<1=欠缴，>1=超额还款 |
| `PAYMENT_DIFF` | `AMT_INSTALMENT - AMT_PAYMENT` | 还款缺口（应还-实还绝对金额）|
| `IS_LATE` | `DPD > 0` | 是否逾期（0/1）|
| `IS_UNDERPAY` | `PAYMENT_DIFF > 0` | 是否少还（0/1）|

**输出特征（全量 + 6M 窗口 + 趋势）**：

| 特征名 | 类型 | 金融含义 |
|--------|------|---------|
| `INST_COUNT` | 全量 | 历史还款记录总条数 |
| `INST_DPD_MEAN/MAX/SUM` | 全量 | 平均/最大/累计逾期天数 |
| `INST_DBD_MEAN` | 全量 | 平均提前还款天数（越大还款习惯越好）|
| `INST_LATE_RATE/COUNT` | 全量 | 逾期率 / 逾期次数 |
| `INST_UNDERPAY_RATE` | 全量 | 少还率 |
| `INST_PAYMENT_RATIO_MEAN/MIN` | 全量 | 平均/最低还款比例 |
| `INST_PAYMENT_DIFF_MEAN/SUM` | 全量 | 平均/累计欠款金额 |
| `INST_RECENT_DPD_MEAN_6M` | 6M 窗口 | 近6月平均逾期天数（`DAYS_INSTALMENT >= -180`）|
| `INST_RECENT_LATE_RATE_6M` | 6M 窗口 | 近6月逾期率 |
| `INST_RECENT_PAY_RATIO_6M` | 6M 窗口 | 近6月平均还款比例 |
| `INST_DPD_TREND` | 趋势 | 近6M均值 / 全量均值，`>1` 表示近期逾期恶化 |
| `INST_LATE_TREND` | 趋势 | 近6M逾期率 / 全量逾期率，方向性预警信号 |

**业务逻辑说明**：近期行为比远期行为更能预测未来违约（行为漂移捕捉）。`INST_DPD_TREND > 1` 是最强的风险预警信号之一，说明客户近期还款能力在恶化。

**调用示例**：
```python
from src.feature_engineering import feat_installments
inst_feat = feat_installments(data['installments'])
```

---

#### `feat_credit_card(cc: pd.DataFrame) -> pd.DataFrame`

**功能描述**：从信用卡月度账单快照中提取信用卡使用行为特征，核心是利用率和还款完整性。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `cc` | pd.DataFrame | `credit_card_balance.csv`，3,840,312 行，每行一张卡某月的账单快照 |

**返回值**：`pd.DataFrame`，以 `SK_ID_CURR` 为主键的聚合特征表

**中间衍生变量**：

| 变量 | 公式 | 金融含义 |
|------|------|---------|
| `CC_UTILIZATION` | `AMT_BALANCE / (AMT_CREDIT_LIMIT_ACTUAL+1)` | 信用卡使用率（FICO 最关键指标之一，>30% 开始扣分，>90% 极高风险）|
| `CC_PAYMENT_RATIO` | `AMT_PAYMENT_CURRENT / (AMT_BALANCE+1)` | 当月还款比例，=1 全额还款；低→滚动负债累积 |
| `CC_MIN_PAY_RATIO` | `AMT_PAYMENT_CURRENT / (AMT_PAYMENT_TOTAL_CURRENT+1)` | 最低还款比率，仅还最低额→偿债意愿弱 |
| `CC_IS_DPD` | `SK_DPD > 0` | 该账单月是否逾期 |

**输出聚合特征**：

| 特征名 | 金融含义 |
|--------|---------|
| `CC_COUNT` | 信用卡账单总条数（使用时长）|
| `CC_BALANCE_MEAN/MAX` | 月末余额均值/最大值，高→持续高负债 |
| `CC_LIMIT_MEAN` | 平均信用额度，高→资质较好 |
| `CC_UTILIZATION_MEAN/MAX` | 使用率均值（日常习惯）/最大值（历史峰值压力）|
| `CC_PAYMENT_RATIO_MEAN` | 平均还款比例 |
| `CC_MIN_PAY_RATIO_MEAN` | 长期仅还最低额是强风险信号 |
| `CC_DPD_MEAN/MAX` | 平均/最大逾期天数 |
| `CC_IS_DPD_RATE` | 逾期账单月比率 |
| `CC_DRAWINGS_MEAN` | 平均月消费金额（`AMT_DRAWINGS_CURRENT`）|

**调用示例**：
```python
from src.feature_engineering import feat_credit_card
cc_feat = feat_credit_card(data['credit_card'])
```

---

#### `feat_pos(pos: pd.DataFrame) -> pd.DataFrame`

**功能描述**：从 POS/现金贷月度快照中提取 HC 内部产品线的还款状态特征，与 `feat_bureau` 的外部数据互补。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `pos` | pd.DataFrame | `POS_CASH_balance.csv`，10,001,358 行，每行一笔 HC 历史贷款某月快照 |

**返回值**：`pd.DataFrame`，以 `SK_ID_CURR` 为主键的聚合特征表

**中间衍生变量**：

| 变量 | 公式 | 金融含义 |
|------|------|---------|
| `POS_IS_DPD` | `SK_DPD > 0` | 普通逾期（宽口径）|
| `POS_IS_DPD_DEF` | `SK_DPD_DEF > 0` | 严重逾期（窄口径，监管定义违约级别）|
| `POS_PROGRESS` | `1 - CNT_INSTALMENT_FUTURE / (CNT_INSTALMENT+1)` | 还款进度，接近1→即将还清仍逾期→持续偿债能力不足 |

**输出聚合特征**：

| 特征名 | 金融含义 |
|--------|---------|
| `POS_COUNT` | 历史 POS/现金贷账单总条数 |
| `POS_DPD_MEAN/MAX` | 平均/最大逾期天数 |
| `POS_IS_DPD_RATE` | 宽口径逾期月比率 |
| `POS_IS_DPD_DEF_RATE` | 窄口径严重逾期月比率（信号更强）|
| `POS_PROGRESS_MEAN` | 平均还款进度（训练结果中 LGB importance 排名 3）|
| `POS_MONTHS_COUNT` | 贷款存续月数（与 HC 关系深度）|

**调用示例**：
```python
from src.feature_engineering import feat_pos
pos_feat = feat_pos(data['pos'])
```

---

#### `feat_previous(prev: pd.DataFrame) -> pd.DataFrame`

**功能描述**：从 HC 历史申请记录中提取申请行为模式特征，类似 FICO 的"新申请"维度（占10%权重）。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `prev` | pd.DataFrame | `previous_application.csv`，1,670,214 行，每行一笔历史申请 |

**返回值**：`pd.DataFrame`，以 `SK_ID_CURR` 为主键的聚合特征表

**中间衍生变量**：

| 变量 | 公式 / 规则 | 含义 |
|------|------------|------|
| `PREV_CREDIT_APP_RATIO` | `AMT_CREDIT / (AMT_APPLICATION+1)` | 实批/申请比，<1 说明被砍额，长期砍额→HC 历史认为风险偏高 |
| `IS_APPROVED` | `NAME_CONTRACT_STATUS == 'Approved'` | 是否被批准 |
| `IS_REFUSED` | `NAME_CONTRACT_STATUS == 'Refused'` | 是否被拒绝（内部信用信号）|
| `IS_CANCELED` | `NAME_CONTRACT_STATUS == 'Canceled'` | 客户主动取消 |
| `DOWN_PAY_RATIO` | `AMT_DOWN_PAYMENT / (AMT_CREDIT+1)` | 首付比例，越高→自有资金越充足→风险越低 |

**输出聚合特征**：

| 特征名 | 金融含义 |
|--------|---------|
| `PREV_COUNT` | 历史申请总笔数 |
| `PREV_APPROVED/REFUSED/CANCELED_COUNT` | 各结果绝对笔数 |
| `PREV_APPROVED_RATE` | HC 内部信用认可度 |
| `PREV_REFUSED_RATE` | HC 历史拒绝率，高→HC 长期认为该客户风险偏高（强负向特征）|
| `PREV_CREDIT_MEAN/MAX` | 历史授信均值/最大值 |
| `PREV_ANNUITY_MEAN` | 历史平均年化还款额 |
| `PREV_DOWN_PAY_RATIO` | 平均首付比例 |
| `PREV_CNT_PAYMENT_MEAN` | 平均合同期数（月）|
| `PREV_DAYS_DECISION_MEAN` | 平均决策距申请天数（越近0→越近期有申请）|

**调用示例**：
```python
from src.feature_engineering import feat_previous
prev_feat = feat_previous(data['previous'])
```

---

#### `feat_cross(df: pd.DataFrame) -> pd.DataFrame`

**功能描述**：在全表 merge 完成后执行，利用多张表的聚合结果构造复合风险信号特征。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `df` | pd.DataFrame | 已合并所有子表聚合特征的宽表 |

**返回值**：`pd.DataFrame`，追加交叉特征列（原地修改副本）

**输出交叉特征**：

| 特征名 | 公式 | 业务逻辑 |
|--------|------|---------|
| `BUREAU_DEBT_INCOME_RATIO` | `BUREAU_DEBT_SUM / (AMT_INCOME_TOTAL+1)` | 外部总负债/年收入，全口径杠杆率（比 CREDIT_INCOME_RATIO 更完整）|
| `REFUSED_RATE_CREDIT_CROSS` | `PREV_REFUSED_RATE × CREDIT_INCOME_RATIO` | 历史被拒 × 当前贷款压力，双重负向叠加 |
| `CC_UTIL_BUREAU_DPD_CROSS` | `CC_UTILIZATION_MEAN × BUREAU_BB_DPD_RATE_MEAN` | 信用卡高利用率 × 外部逾期率，流动性紧张+不良习惯复合信号 |
| `EMPLOYED_EXT2_CROSS` | `DAYS_EMPLOYED_YEARS × EXT_SOURCE_2` | 就业稳定 × 外部好评分，优质客群正向放大；反之高风险 |

**调用示例**：
```python
from src.feature_engineering import feat_cross
df = feat_cross(df_merged)
```

---

#### `build_features(data: dict) -> pd.DataFrame`

**功能描述**：特征工程主入口，按顺序调用所有子函数，最终输出融合后的完整宽表。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `data` | dict | `load_all()` 的返回值（7张表的字典）|

**返回值**：`pd.DataFrame`，shape=(307511, 214)，含 `SK_ID_CURR`、`TARGET` 及所有衍生特征

**内部执行顺序**：
```
1. clean_application(app)          → 主表清洗
2. feat_application(app)           → 主表衍生特征
3. feat_bureau(bureau, bureau_bal) → 外部征信聚合
4. feat_installments(inst)         → 分期还款聚合
5. feat_credit_card(cc)            → 信用卡行为聚合
6. feat_pos(pos)                   → POS 行为聚合
7. feat_previous(prev)             → 历史申请聚合
8. 所有子表特征 left join 到主表
9. feat_cross(df)                  → 跨表交叉特征
10. encode_categoricals(df)        → 类别变量编码
11. fill_missing(df)               → 缺失值填充
```

**调用示例**：
```python
from src.feature_engineering import build_features
df = build_features(data)   # 耗时约 3~5 分钟，建议缓存到 features.pkl
df.to_pickle('data/processed/features.pkl')
```

---

### 2.5 `src/feature_selection.py`

**模块职责**：使用 LightGBM 5 折交叉验证评估全量特征重要性，筛选出 Top-N 特征供 WoE 编码使用。

**依赖模块**：`src/config`（`LGB_PARAMS`, `LGB_CV_FOLDS`, `LGB_EARLY_STOP`, `TARGET`, `ID_COL`, `MODEL_DIR`, `FEATURE_DIR`）

**对外提供的函数**：`run_lgb_cv(df)`、`select_features(imp_df, df, top_n, threshold)`

---

#### `run_lgb_cv(df: pd.DataFrame) -> tuple`

**功能描述**：对全量特征做 LightGBM 5 折交叉验证，返回 OOF 预测值和特征重要性排名。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `df` | pd.DataFrame | `build_features` 输出的完整特征宽表（含 `TARGET`）|

**返回值**：

| 返回项 | 类型 | 含义 |
|--------|------|------|
| `oof_pred` | np.ndarray | shape=(307511,)，OOF 预测违约概率 |
| `imp_df` | pd.DataFrame | 各特征在 5 折中的平均重要性，按降序排列，列：`[feature, importance]` |

**副作用（保存文件）**：
- `outputs/models/lgb_fold{1-5}.pkl`：各折 LGB 模型
- `outputs/features/lgb_importance.csv`：特征重要性排名

**业务逻辑说明**：此步骤不追求极致 AUC，目的是获得**稳定的特征重要性排名**。通过 5 折取均值、强正则、高 `min_child_samples` 抑制噪声特征"虚高"。实际 OOF AUC=0.7881。

**调用示例**：
```python
from src.feature_selection import run_lgb_cv
oof_pred, imp_df = run_lgb_cv(df)
print(imp_df.head(20))
```

---

#### `select_features(imp_df, df, top_n=None, threshold=None) -> list`

**功能描述**：根据 LGB 重要性排名筛选特征，优先使用 `top_n`。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `imp_df` | pd.DataFrame | `run_lgb_cv` 返回的特征重要性表 |
| `df` | pd.DataFrame | 特征宽表（用于确认特征存在性）|
| `top_n` | int \| None | 取重要性前 N 名，优先使用（默认 `TARGET_FEATURE_NUM=40`）|
| `threshold` | float \| None | 按重要性绝对值筛选（当 `top_n=None` 时生效）|

**返回值**：`list`，筛选后的特征名列表（40 个）

**副作用**：保存 `outputs/features/selected_features.csv`

**调用示例**：
```python
from src.feature_selection import select_features
selected = select_features(imp_df, df, top_n=40)
# ['CREDIT_TERM', 'EXT_SOURCE_MEAN', 'POS_PROGRESS_MEAN', ...]
```

---

### 2.6 `src/woe_encoding.py`

**模块职责**：对 LGB 筛选后的特征做 WoE/IV 计算、单调性约束、IV 过滤和编码转换，为 LR 建模准备线性可分输入。

**依赖模块**：`src/config`（`FEATURE_DIR`, `SCORECARD_DIR`, `TARGET`）

**对外提供的函数**：`bin_numeric`、`bin_categorical`、`calculate_woe_iv`、`enforce_monotone`、`fit_woe`、`transform_woe`、`save_woe`、`load_woe`、`run_woe_encoding`

**关键常量**：

| 常量 | 值 | 含义 |
|------|----|------|
| `IV_THRESHOLD` | 0.02 | 低于此值视为无预测力，剔除 |
| `MAX_BINS` | 10 | 数值型最大等频分箱数 |
| `MIN_BIN_RATE` | 0.05 | 单箱最小样本占比（防止 WoE 无穷大）|

---

#### `bin_numeric(x, y, max_bins=10, min_bin_rate=0.05) -> np.ndarray`

**功能描述**：对数值型特征做等频分箱，迭代合并过小箱/纯好/纯坏箱，返回分箱边界数组。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `x` | pd.Series | 特征列 |
| `y` | pd.Series | TARGET 列 |
| `max_bins` | int | 最大分箱数，默认 10 |
| `min_bin_rate` | float | 单箱最小样本占比，默认 0.05 |

**返回值**：`np.ndarray`，分箱边界（首尾为 `-inf`/`+inf`）

**业务逻辑说明**：等频分箱保证每箱有足够统计量。合并条件：① 箱样本占比 < `min_bin_rate`；② 纯好人箱；③ 纯坏人箱。以上三种情况均无法计算有效 WoE（分母为0）。

---

#### `bin_categorical(x, y) -> dict`

**功能描述**：对类别型特征统计各类别频率，将低频类别（<`MIN_BIN_RATE`）合并为 `"OTHER"`。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `x` | pd.Series | 类别型特征列 |
| `y` | pd.Series | TARGET 列 |

**返回值**：`dict`，`{原始类别: 合并后标签}`，低频类均映射到 `"OTHER"`

---

#### `calculate_woe_iv(x, y, bins=None, cat_map=None, is_categorical=False) -> pd.DataFrame`

**功能描述**：给定分箱方案，计算每个箱的 WoE 和 IV，缺失值单独作为 `NaN` 箱处理。

**WoE / IV 公式**：

$$\text{WoE}_i = \ln\left(\frac{\text{好人}_i / \text{总好人}}{\text{坏人}_i / \text{总坏人}}\right)$$

$$\text{IV}_i = \left(\frac{\text{好人}_i}{\text{总好人}} - \frac{\text{坏人}_i}{\text{总坏人}}\right) \times \text{WoE}_i$$

$$\text{总IV} = \sum_i \text{IV}_i \quad (\text{NaN 箱不计入})$$

**WoE 业务含义**：
- `WoE > 0`：该箱好人占比 > 坏人占比（低风险区间）
- `WoE < 0`：该箱坏人占比 > 好人占比（高风险区间）
- 绝对值越大，该箱区分度越强

> 使用 Laplace 平滑（加 0.5）防止 0/0 问题。

**返回值**：`pd.DataFrame`，列：`[bin_label, count, bad, good, bad_rate, bad_pct, good_pct, woe, iv]`

---

#### `enforce_monotone(x, y, bins) -> np.ndarray`

**功能描述**：通过逐步合并相邻分箱来强制 WoE 序列单调（递增或递减）。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `x` | pd.Series | 特征列 |
| `y` | pd.Series | TARGET 列 |
| `bins` | np.ndarray | `bin_numeric` 返回的初始边界 |

**返回值**：`np.ndarray`，满足单调性的最终分箱边界

**业务逻辑说明**：监管/业务合规要求——特征与违约概率的关系必须单调，否则评分卡在某些分段的逻辑会自相矛盾（如"收入越高违约率先降后升"是无法向业务解释的）。算法找出第一个违反主方向的相邻箱对并合并，迭代直到满足单调。

---

#### `fit_woe(df, feature_cols, target_col=TARGET) -> tuple`

**功能描述**：对所有选定特征逐一拟合 WoE 编码方案，输出映射字典和 IV 汇总表。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `df` | pd.DataFrame | 含特征和 TARGET 的完整宽表 |
| `feature_cols` | list | LGB 筛选后的特征列表（40个）|
| `target_col` | str | 目标变量列名，默认 `'TARGET'` |

**返回值**：

| 返回项 | 类型 | 含义 |
|--------|------|------|
| `woe_map` | dict | `{特征名: woe_df}`，每个特征的分箱→WoE映射表 |
| `iv_summary` | pd.DataFrame | 列：`[feature, iv, n_bins, keep]`，`keep=True` 表示 IV≥0.02 |

**处理流程**（每个特征）：
```
判断类型（数值/类别）
  ↓
数值型：bin_numeric → enforce_monotone → calculate_woe_iv
类别型：bin_categorical → calculate_woe_iv
  ↓
累加 IV，标记 keep（IV >= 0.02）
```

实际结果：40 个候选特征 → 保留 28 个（IV<0.02 的 12 个剔除）

---

#### `transform_woe(df, woe_map, iv_summary) -> pd.DataFrame`

**功能描述**：将原始特征替换为 WoE 值，返回 LR 可直接使用的编码矩阵。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `df` | pd.DataFrame | 原始特征宽表 |
| `woe_map` | dict | `fit_woe` 的返回值 |
| `iv_summary` | pd.DataFrame | `fit_woe` 的返回值，用于过滤 `keep=False` 的特征 |

**返回值**：`pd.DataFrame`，shape=(307511, 30)，列名格式为 `{原始特征名}_WOE`，另含 `SK_ID_CURR`、`TARGET`

**业务逻辑说明**：WoE 编码后每个箱用一个实数表示，与 log-odds 天然线性，满足 LR 的线性假设。未知类别（生产新类别）自动 fallback 到 `OTHER` 箱的 WoE；缺失值 fallback 到 `NaN` 箱的 WoE，不会产生新缺失。

---

#### `save_woe(woe_map, iv_summary) -> None` / `load_woe() -> tuple`

**功能描述**：将 WoE 映射表和 IV 汇总持久化到 CSV；`load_woe` 反向重建字典（推理/复盘用）。

**输出文件**：
- `outputs/features/woe_map.csv`
- `outputs/features/iv_summary.csv`

---

#### `run_woe_encoding(df, selected_features) -> tuple`

**功能描述**：WoE 编码完整流程入口（`main.py` Step 5 调用）。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `df` | pd.DataFrame | `build_features` 输出的宽表 |
| `selected_features` | list | `select_features` 返回的 40 个特征名 |

**返回值**：

| 返回项 | 类型 | 含义 |
|--------|------|------|
| `df_woe` | pd.DataFrame | WoE 编码矩阵（307511 × 30）|
| `woe_map` | dict | 分箱映射字典（供 `scorecard.py` 使用）|
| `iv_summary` | pd.DataFrame | IV 汇总表（供 `model_lr.py` 使用）|

**调用示例**：
```python
from src.woe_encoding import run_woe_encoding
df_woe, woe_map, iv_summary = run_woe_encoding(df, selected_features)
```

---

### 2.7 `src/model_lr.py`

**模块职责**：在 WoE 编码矩阵上完成共线性剔除、LR 5 折交叉验证训练、系数稳定性评估和方向检验。

**依赖模块**：`src/config`（`TARGET`, `ID_COL`, `CORRELATION_THRESHOLD`, `LGB_CV_FOLDS`, `MODEL_DIR`, `FEATURE_DIR`）

**对外提供的函数**：`remove_collinear`、`calc_ks`、`train_lr_cv`、`build_coef_table`、`run_lr_modeling`

---

#### `remove_collinear(df_woe, iv_summary, threshold=0.85) -> list`

**功能描述**：基于 Pearson 相关系数矩阵剔除 WoE 特征中的共线性对，保留 IV 更高者。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `df_woe` | pd.DataFrame | WoE 编码矩阵 |
| `iv_summary` | pd.DataFrame | IV 汇总表（用于比较保留哪个）|
| `threshold` | float | 相关系数阈值，默认 0.85 |

**返回值**：`list`，共线性处理后保留的 WoE 特征列名（实际：28→23个）

**业务逻辑说明**：LR 对多重共线性敏感——相关系数过高的两个特征会导致系数符号反转、标准误爆炸，破坏评分卡的业务可解释性。`0.85` 是行业经验阈值，允许适度相关（如 `DAYS_BIRTH` 和 `AGE_YEARS` 完全相关 corr=1.0，必须剔除其一）。

**副作用**：保存 `outputs/features/lr_features.csv`

---

#### `calc_ks(y_true, y_prob) -> float`

**功能描述**：计算 KS 统计量（Kolmogorov-Smirnov）。

**公式**：

$$\text{KS} = \max_t \left| \text{TPR}(t) - \text{FPR}(t) \right| = \max_t \left( \text{坏人累积占比}(t) - \text{好人累积占比}(t) \right)$$

**KS 行业参考标准**：

| KS 范围 | 评级 |
|---------|------|
| < 0.20 | 较弱，需重审特征 |
| 0.20 ~ 0.30 | 可接受 |
| 0.30 ~ 0.40 | 良好 ✓ |
| > 0.40 | 优秀 |
| > 0.75 | 可能过拟合 |

本项目 OOF KS = **0.3663**（良好）

---

#### `train_lr_cv(df_woe, feature_cols) -> tuple`

**功能描述**：LR 5 折交叉验证训练，每折返回 `(StandardScaler, LogisticRegression)` 元组并保存到磁盘。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `df_woe` | pd.DataFrame | WoE 编码矩阵（含 TARGET）|
| `feature_cols` | list | 共线性处理后保留的 WoE 特征列 |

**返回值**：

| 返回项 | 类型 | 含义 |
|--------|------|------|
| `oof_pred` | np.ndarray | OOF 预测违约概率 |
| `models` | list | 5 个 `(scaler, lr)` 元组 |
| `cv_metrics` | pd.DataFrame | 各折 AUC/KS/Gini 及 OOF 汇总 |

**关键参数说明**：

| 参数 | 值 | 原因 |
|------|----|------|
| `C` | 0.1 | C=1/λ，值越小正则越强；评分卡偏保守，防系数过大 |
| `solver` | `'lbfgs'` | 拟牛顿法，适合中等规模（<30万×50特征），支持 L2 正则 |
| `class_weight` | `'balanced'` | 自动处理 1:11 样本不平衡，等价于 LGB 的 `scale_pos_weight=11` |
| `StandardScaler` | 训练集 fit | 标准化后系数绝对值可直接比较各特征影响力；避免数据泄露 |

**副作用**：保存 `outputs/models/lr_fold{1-5}.pkl`、`outputs/models/lr_cv_metrics.csv`

---

#### `build_coef_table(models, feature_cols, iv_summary) -> pd.DataFrame`

**功能描述**：汇总各折 LR 系数，计算稳定性指标，执行方向检验，提取截距行。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `models` | list | `train_lr_cv` 返回的 `(scaler, lr)` 列表 |
| `feature_cols` | list | WoE 特征列名 |
| `iv_summary` | pd.DataFrame | IV 汇总（用于排序和附加 IV 信息）|

**返回值**：`pd.DataFrame`，列：`[feature, coef_mean, coef_std, coef_min, coef_max, coef_fold1~5, iv, direction_ok, direction, coef_cv]`，末尾含 `_INTERCEPT` 行

**方向检验原则**：

> WoE 编码后，LR 系数理论上应全为负数：
> - WoE 越高 → 好人比例越高 → 违约概率越低
> - 因此：`log-odds = β₀ + β₁×WoE` 中，`β₁ < 0` 才符合业务逻辑
> - 若 `β₁ > 0`，说明该特征存在 WoE 单调性问题或共线性残留

本项目发现 3 个方向异常特征：`EXT_SOURCE_2_AGE_WOE`（与 `EXT_SOURCE_2` 高度共线）、`DAYS_LAST_PHONE_CHANGE_WOE`（系数≈0）、`AMT_GOODS_PRICE_WOE`（轻微异常）。

**截距处理**：末尾附加 `_INTERCEPT` 行，存储 5 折截距均值，供 `build_scorecard_table` 均摊到每个特征的基础分。

**副作用**：保存 `outputs/models/lr_coef_table.csv`

---

#### `run_lr_modeling(df_woe, iv_summary) -> tuple`

**功能描述**：LR 建模完整流程入口（`main.py` Step 6 调用）。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `df_woe` | pd.DataFrame | `run_woe_encoding` 返回的 WoE 矩阵 |
| `iv_summary` | pd.DataFrame | `run_woe_encoding` 返回的 IV 汇总 |

**返回值**：

| 返回项 | 类型 | 含义 |
|--------|------|------|
| `models` | list | 5 折 `(scaler, lr)` 元组列表（供评分卡使用）|
| `feature_cols` | list | 最终进入模型的 23 个 WoE 特征列名 |
| `coef_df` | pd.DataFrame | 系数稳定性表（含 `_INTERCEPT` 行）|
| `cv_metrics` | pd.DataFrame | 各折 AUC/KS/Gini |

**调用示例**：
```python
from src.model_lr import run_lr_modeling
models, lr_features, coef_df, cv_metrics = run_lr_modeling(df_woe, iv_summary)
```

---

### 2.8 `src/scorecard.py`

**模块职责**：将 LR 系数和 WoE 映射表转换为标准评分卡格式，提供单人评分函数和批量校验。

**依赖模块**：`src/config`（`PDO`, `BASE_SCORE`, `BASE_ODDS`, `SCORECARD_DIR`, `TARGET`, `ID_COL`）

**对外提供的函数**：`calc_scorecard_params`、`build_scorecard_table`、`score_single`、`validate_scorecard`、`save_scorecard`、`print_scorecard`、`run_scorecard`

---

#### `calc_scorecard_params(pdo=20, base_score=600, base_odds=50) -> tuple`

**功能描述**：计算评分卡转换所需的 `factor` 和 `offset` 两个核心参数。

**参数说明**：

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `pdo` | float | 20 | Points to Double Odds：好坏比每翻倍，分值增加 PDO 分 |
| `base_score` | float | 600 | 基准分（好坏比等于 `base_odds` 时对应的分值）|
| `base_odds` | float | 50 | 基准好坏比（好人数/坏人数）|

**返回值**：`(factor, offset)`，均为 float

**完整推导过程**：

设评分卡得分公式为：

$$\text{score} = \text{offset} - \text{factor} \times \ln(\text{odds})$$

其中 $\text{odds} = P(\text{好人}) / P(\text{坏人})$，与 LR 的 log-odds 对应。

**由 PDO 定义**（好坏比翻倍时分数增加 PDO 分）：

$$\text{score} + \text{PDO} = \text{offset} - \text{factor} \times \ln(2 \times \text{odds})$$

两式相减：

$$\text{PDO} = \text{factor} \times \ln 2 \implies \boxed{\text{factor} = \frac{\text{PDO}}{\ln 2}}$$

**由基准分定义**（好坏比为 `base_odds` 时分值为 `base_score`）：

$$\text{base\_score} = \text{offset} - \text{factor} \times \ln(\text{base\_odds})$$

$$\implies \boxed{\text{offset} = \text{base\_score} + \text{factor} \times \ln(\text{base\_odds})}$$

**本项目计算结果**：

$$\text{factor} = \frac{20}{\ln 2} \approx 28.8539, \quad \text{offset} = 600 + 28.8539 \times \ln 50 \approx 712.8771$$

---

#### `build_scorecard_table(woe_map, coef_df, factor, offset) -> pd.DataFrame`

**功能描述**：将 LR 系数 × WoE 映射表转换为每个分箱的标准得分。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `woe_map` | dict | `fit_woe` 返回的分箱映射字典 |
| `coef_df` | pd.DataFrame | `build_coef_table` 返回的系数表（含 `_INTERCEPT` 行）|
| `factor` | float | `calc_scorecard_params` 返回值 |
| `offset` | float | `calc_scorecard_params` 返回值 |

**返回值**：`pd.DataFrame`，列：`[feature, bin_label, woe, coef, score, count, bad_rate, is_nan_bin]`

**每箱得分公式**：

LR 模型展开：

$$\log\text{-odds} = \beta_0 + \sum_{j=1}^{n} \beta_j \times \text{WoE}_{j}$$

将截距均摊到每个特征（每个特征分担 $\beta_0 / n$ 的基础分）：

$$\boxed{\text{score}_{ji} = -\left(\beta_j \times \text{WoE}_{ji} + \frac{\beta_0}{n}\right) \times \text{factor} + \frac{\text{offset}}{n}}$$

**负号的来源**：score 越高 → odds 越大 → 好人占比越高 → 违约概率越低，而 LR 中 log-odds 越大表示违约概率越低，因此需要负号对齐方向。

**总分**：

$$\text{total\_score} = \sum_{j=1}^{n} \text{score}_{j,\text{bin}(j)} = -\text{factor} \times \log\text{-odds} + \text{offset}$$

---

#### `score_single(applicant, scorecard, woe_map) -> dict`

**功能描述**：给定单个申请人的原始特征字典，查评分卡返回总分和各特征明细。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `applicant` | dict | `{原始特征名: 特征值}`（不含 `_WOE` 后缀）|
| `scorecard` | pd.DataFrame | `build_scorecard_table` 生成的评分卡表 |
| `woe_map` | dict | WoE 映射字典（用于查找分箱）|

**返回值**：
```python
{
    'total_score': 735.5,          # 总分（越高越像好人）
    'details': [
        {'feature': 'EXT_SOURCE_MEAN', 'raw_value': 0.65,
         'bin_label': '(0.644, 0.692]', 'score': 44.9},
        ...
    ]
}
```

**分箱查找规则**：
- `None` 或 `NaN` → `'NaN'` 箱
- 类别型 → 直接匹配 `bin_label`；未知类别 → `'OTHER'` 箱
- 数值型 → 解析区间边界，用 `pd.cut` 定位所在区间

---

#### `validate_scorecard(scorecard, df_raw, woe_map) -> pd.DataFrame`

**功能描述**：批量评分（最多 5000 行采样），校验评分卡方向性和分段违约率单调性。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `scorecard` | pd.DataFrame | 评分卡表 |
| `df_raw` | pd.DataFrame | 含原始特征和 TARGET 的宽表 |
| `woe_map` | dict | WoE 映射字典 |

**返回值**：`pd.DataFrame`，列：`[score, TARGET]`

**校验项目**：
1. 分数分布（均值/标准差/min/max）
2. 好坏人均分对比（好人均分 > 坏人均分 = 方向正确）
3. 10 分位分段违约率（应单调递减）

本项目结果：好人均分 729.5，坏人均分 716.9，差值 12.7 分，方向 ✅

> `MAX_VALIDATE_ROWS=5000` 采样限制，避免 30 万行 Python 逐行循环耗时数小时。

---

#### `run_scorecard(df_raw, woe_map, coef_df, iv_summary, validate=True) -> pd.DataFrame`

**功能描述**：评分卡生成完整流程入口（`main.py` Step 7 调用）。

**参数说明**：

| 参数 | 类型 | 含义 |
|------|------|------|
| `df_raw` | pd.DataFrame | 含原始特征和 TARGET 的宽表（用于校验）|
| `woe_map` | dict | `run_woe_encoding` 返回值 |
| `coef_df` | pd.DataFrame | `run_lr_modeling` 返回值 |
| `iv_summary` | pd.DataFrame | `run_woe_encoding` 返回值 |
| `validate` | bool | 是否执行批量校验，数据量大时可设 `False` 跳过 |

**返回值**：`pd.DataFrame`，标准评分卡（156 条分箱记录）

**副作用**：保存 `outputs/scorecard/scorecard.csv` 和 `outputs/scorecard/score_dist.csv`

**调用示例**：
```python
from src.scorecard import run_scorecard
scorecard = run_scorecard(
    df_raw=df, woe_map=woe_map,
    coef_df=coef_df, iv_summary=iv_summary,
    validate=True
)
```

---

### 2.9 `main.py`

**模块职责**：全流程编排入口，按顺序调用 7 个步骤，支持特征工程缓存（第二次运行跳过重建）。

**依赖模块**：全部 `src/` 模块

**运行方式**：
```bash
cd /Users/sanchuankanzi/Documents/code/home_credit_scorecard
python main.py
# 后台运行（推荐，耗时约30~60分钟）：
nohup python -u main.py > outputs/run_log.txt 2>&1 &
```

---

## 3. 完整调用链路

```
main()
│
├── Step 1  load_all()
│           └── load_data(name) × 7          →  data: dict{7 DataFrames}
│
├── Step 2  build_features(data)              →  df: (307511, 214)
│           ├── clean_application(app)
│           ├── feat_application(app)
│           ├── feat_bureau(bureau, bureau_bal)
│           ├── feat_installments(inst)
│           ├── feat_credit_card(cc)
│           ├── feat_pos(pos)
│           ├── feat_previous(prev)
│           ├── [left join 所有子表到主表]
│           ├── feat_cross(df)
│           ├── encode_categoricals(df)
│           └── fill_missing(df)
│           ↓ 缓存到 data/processed/features.pkl
│
├── Step 3  run_lgb_cv(df)                   →  oof_pred, imp_df
│           └── LGBMClassifier × 5折
│               ├── 每折保存 lgb_fold{i}.pkl
│               └── 保存 lgb_importance.csv
│
├── Step 4  select_features(imp_df, df, top_n=40)  →  selected: list[40]
│           └── 保存 selected_features.csv
│
├── Step 5  run_woe_encoding(df, selected)   →  df_woe, woe_map, iv_summary
│           ├── fit_woe(df, selected)
│           │   ├── bin_numeric / bin_categorical（每个特征）
│           │   ├── enforce_monotone（数值型）
│           │   └── calculate_woe_iv
│           ├── save_woe → woe_map.csv, iv_summary.csv
│           └── transform_woe → df_woe: (307511, 30)
│           [40特征 → IV<0.02剔除12个 → 保留28个]
│
├── Step 6  run_lr_modeling(df_woe, iv_summary)  →  models, lr_features, coef_df, cv_metrics
│           ├── remove_collinear → 28→23个特征，保存 lr_features.csv
│           ├── train_lr_cv → 5折 (scaler, lr)，保存 lr_fold{i}.pkl, lr_cv_metrics.csv
│           └── build_coef_table → 系数表+截距行，保存 lr_coef_table.csv
│
└── Step 7  run_scorecard(df, woe_map, coef_df, iv_summary)  →  scorecard
            ├── calc_scorecard_params() → factor=28.85, offset=712.88
            ├── build_scorecard_table → 156条分箱得分记录
            ├── validate_scorecard（采样5000行校验）
            │   └── score_single × 5000
            ├── save_scorecard → scorecard.csv, score_dist.csv
            └── print_scorecard（打印前5个特征）
```

**数据流转一览**：

| Step | 输入 | 输出 | shape 变化 |
|------|------|------|-----------|
| 1 数据加载 | 7个CSV | `data` dict | — |
| 2 特征工程 | `data` | `df` | → (307511, 214) |
| 3 LGB筛选 | `df` | `imp_df` | 212特征排序 |
| 4 Top-N筛选 | `imp_df` | `selected` | 212 → 40 |
| 5 WoE编码 | `df` + `selected` | `df_woe` | 40 → 28 → (307511, 30) |
| 6 LR建模 | `df_woe` | `models`, `coef_df` | 28 → 23特征 |
| 7 评分卡 | `woe_map`+`coef_df` | `scorecard` | 23特征 × N箱 = 156条 |

---

## 4. 输入输出文件说明

### 4.1 输入文件（`data/raw/`）

| 文件名 | 主键 | shape | 说明 |
|--------|------|-------|------|
| `application_train.csv` | `SK_ID_CURR` | (307511, 122) | 主表：申请人基本信息、贷款信息、目标变量 |
| `bureau.csv` | `SK_ID_BUREAU` | (1716428, 17) | 外部征信：每笔外部贷款的基本信息 |
| `bureau_balance.csv` | `SK_ID_BUREAU` | (27299925, 3) | 外部征信月度还款状态快照 |
| `previous_application.csv` | `SK_ID_PREV` | (1670214, 37) | HC 历史申请记录 |
| `POS_CASH_balance.csv` | `SK_ID_PREV` | (10001358, 8) | HC POS/现金贷月度还款快照 |
| `credit_card_balance.csv` | `SK_ID_PREV` | (3840312, 23) | HC 信用卡月度账单快照 |
| `installments_payments.csv` | `SK_ID_PREV` | (13605401, 8) | HC 分期还款流水 |

### 4.2 中间产物（`data/processed/`）

| 文件名 | 格式 | 说明 |
|--------|------|------|
| `features.pkl` | pickle | `build_features` 输出的完整特征宽表（307511×214），用于避免重复运行特征工程 |

### 4.3 输出文件（`outputs/`）

#### `outputs/models/`

| 文件名 | 说明 |
|--------|------|
| `lgb_fold{1-5}.pkl` | LightGBM 各折模型（joblib 格式）|
| `lr_fold{1-5}.pkl` | LR 各折 `(StandardScaler, LogisticRegression)` 元组 |
| `lr_cv_metrics.csv` | LR 各折 AUC/KS/Gini + OOF 汇总 |
| `lr_coef_table.csv` | LR 系数稳定性表（含各折系数、均值、方向检验、`_INTERCEPT` 行）|

#### `outputs/features/`

| 文件名 | 说明 |
|--------|------|
| `lgb_importance.csv` | LGB 特征重要性排名（feature, importance）|
| `selected_features.csv` | LGB 筛选后的 Top-40 特征名列表 |
| `woe_map.csv` | 所有特征的 WoE 分箱映射表（bin_label, woe, iv, feature 等）|
| `iv_summary.csv` | 各特征 IV 汇总（feature, iv, n_bins, keep）|
| `lr_features.csv` | 共线性处理后保留的 23 个 WoE 特征列名 |

#### `outputs/scorecard/`

| 文件名 | 说明 |
|--------|------|
| `scorecard.csv` | **最终评分卡**，列：feature/bin_label/woe/coef/score/count/bad_rate/is_nan_bin |
| `score_dist.csv` | 训练集（采样5000行）的评分分布，列：score/TARGET，用于分段违约率分析 |

### 4.4 日志文件

| 文件名 | 说明 |
|--------|------|
| `outputs/run_log.txt` | `nohup` 后台运行时的完整训练日志，含各步骤进度、各折 AUC/KS、IV 汇总、系数表等 |

---

*文档生成时间：2026-03-02 | 基于实际训练结果（OOF AUC=0.7456，KS=0.3663，评分范围693~763）*
