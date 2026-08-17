import os

# ── 路径配置 ──────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW       = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUT_DIR     = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR      = os.path.join(OUTPUT_DIR, 'models')
FEATURE_DIR    = os.path.join(OUTPUT_DIR, 'features')
SCORECARD_DIR  = os.path.join(OUTPUT_DIR, 'scorecard')

# 自动创建目录
for _dir in [DATA_PROCESSED, MODEL_DIR, FEATURE_DIR, SCORECARD_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ── 数据文件名 ────────────────────────────────────────────────
FILES = {
    'application'    : 'application_train.csv',
    'bureau'         : 'bureau.csv',
    'bureau_balance' : 'bureau_balance.csv',
    'previous'       : 'previous_application.csv',
    'pos'            : 'POS_CASH_balance.csv',
    'credit_card'    : 'credit_card_balance.csv',
    'installments'   : 'installments_payments.csv',
}

# ── 目标变量 ──────────────────────────────────────────────────
TARGET = 'TARGET'
ID_COL = 'SK_ID_CURR'

# ── LGB 参数 ──────────────────────────────────────────────────
# 调参目标：特征筛选，不是最优AUC
# 核心原则：防过拟合 → 保证各折 feature importance 排名稳定
LGB_PARAMS = {
    'objective'         : 'binary',
    'metric'            : 'auc',
    'boosting_type'     : 'gbdt',

    # 树的数量：配合早停使用，设大一些留足迭代空间
    # 早停会自动找最优轮数，不会真的跑满3000轮
    'n_estimators'      : 3000,

    # 学习率：0.05 是特征筛选阶段的合理默认值
    # 不需要调小（不追求极致AUC），早停已做轮数控制
    'learning_rate'     : 0.05,

    # 叶节点数：31 = 2^5，控制单棵树复杂度
    # 特征筛选阶段无需大树，过复杂的树会过拟合噪声特征拉高其 importance
    'num_leaves'        : 31,

    # 最大深度：限制为6层，配合 num_leaves 双重约束树复杂度
    # -1（无限制）在高维特征下容易过拟合，导致 importance 不稳定
    'max_depth'         : 6,

    # 叶节点最少样本数：307k 数据集建议 ≥ 100
    # 原值20过小 → 树会分出噪声叶节点 → 噪声特征 importance 虚高
    # 100 确保每个叶节点都有足够统计量，importance 更可信
    'min_child_samples' : 100,

    # 叶节点最小分裂增益：过滤 AUC 提升极微小的无效分裂
    # 避免模型在噪声特征上做微小分裂，压低噪声特征的 importance
    'min_split_gain'    : 0.01,

    # 列采样（每棵树）：从 0.8 降至 0.7
    # 更低的列采样 → 每棵树看到的特征更少 → 减少共线性特征"轮流上榜"
    # 有助于在多折中识别出真正稳定重要的特征
    'feature_fraction'  : 0.7,

    # 行采样：保持 0.8，防过拟合的同时保留足够样本多样性
    'bagging_fraction'  : 0.8,
    'bagging_freq'      : 5,

    # L1 正则（稀疏化）：从 0.1 升至 0.5
    # 更强的 L1 让弱特征的分裂权重趋向0，提升重要特征与噪声特征的区分度
    'reg_alpha'         : 0.5,

    # L2 正则（平滑化）：从 0.1 升至 1.0
    # 抑制单棵树对局部噪声的过度拟合，使 importance 跨折更稳定
    'reg_lambda'        : 1.0,

    # 样本不平衡处理：正负比 ≈ 1:11（8.1% 违约）
    # 不加此参数 → 模型偏向多数类 → 少数类特征的 importance 被压制
    # scale_pos_weight = 负样本数 / 正样本数 ≈ 0.919 / 0.081 ≈ 11
    'scale_pos_weight'  : 11,

    'n_jobs'            : -1,
    'random_state'      : 42,
    'verbose'           : -1,
}

LGB_CV_FOLDS   = 5    # K折数
LGB_EARLY_STOP = 150  # 早停窗口：配合更强正则化，给模型更多机会找到真实最优点

# ── 特征筛选参数 ──────────────────────────────────────────────
# LGB importance 阈值：低于此值的特征直接剔除
IMPORTANCE_THRESHOLD   = 10
# 特征筛选后目标数量（供LR使用）
TARGET_FEATURE_NUM     = 40
# 相关系数阈值（共线性剔除）
CORRELATION_THRESHOLD  = 0.85

# ── 评分卡参数 ────────────────────────────────────────────────
# PDO(Points to Double Odds)：每增加PDO分，好坏比翻倍
PDO    = 20
# 基准分
BASE_SCORE = 600
# 基准好坏比
BASE_ODDS  = 50