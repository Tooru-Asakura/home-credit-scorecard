"""
逻辑回归建模模块
流程：共线性剔除 → LR训练（5折CV）→ 模型评估 → 系数表输出
"""
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.config import (
    TARGET, ID_COL,
    CORRELATION_THRESHOLD, LGB_CV_FOLDS,
    MODEL_DIR, FEATURE_DIR,
)


# ════════════════════════════════════════════════════════════
#  Part 1：共线性剔除
# ════════════════════════════════════════════════════════════

def remove_collinear(df_woe: pd.DataFrame,
                     iv_summary: pd.DataFrame,
                     threshold: float = CORRELATION_THRESHOLD) -> list:
    """
    基于 Pearson 相关系数矩阵剔除共线性特征。

    策略：
    - 遍历所有特征对，找出 |corr| > threshold 的配对
    - 两者中保留 IV 更高的特征，剔除 IV 较低的
    - 迭代直到没有超阈值特征对

    业务背景：
    - LR 对多重共线性敏感：高度相关的两个特征会导致系数不稳定
      （系数符号可能反转、标准误爆炸），影响评分卡的业务可解释性
    - 评分卡监管要求每个特征的系数方向必须符合业务逻辑
    - 阈值 0.85 是行业经验值：低于此值允许适度相关，高于则必须剔除

    参数
    ----
    df_woe     : WoE 编码后的特征矩阵（列名为 FEAT_WOE 格式）
    iv_summary : fit_woe 返回的 IV 汇总，含 feature 和 iv 列
    threshold  : 相关系数绝对值阈值，默认 0.85

    返回
    ----
    keep_cols : 剔除共线性后保留的 WoE 特征列名列表
    """
    # 提取 WoE 特征列（排除 ID / TARGET）
    woe_cols = [c for c in df_woe.columns
                if c not in (ID_COL, TARGET)]

    if len(woe_cols) == 0:
        print("[共线性] 无可用特征，跳过")
        return woe_cols

    # 构建 IV 查找字典：WoE列名 → IV值
    # WoE 列名格式为 原始特征名_WOE，需要反查 iv_summary
    iv_dict = {}
    for woe_col in woe_cols:
        orig = woe_col.replace('_WOE', '')
        iv_val = iv_summary.loc[
            iv_summary['feature'] == orig, 'iv'
        ]
        iv_dict[woe_col] = float(iv_val.values[0]) if len(iv_val) > 0 else 0.0

    # 计算相关系数矩阵
    corr_matrix = df_woe[woe_cols].corr().abs()

    keep  = set(woe_cols)
    drop  = set()

    # 上三角遍历，避免重复检查
    for i in range(len(woe_cols)):
        for j in range(i + 1, len(woe_cols)):
            col_i = woe_cols[i]
            col_j = woe_cols[j]

            # 如果其中一个已被标记剔除，跳过
            if col_i in drop or col_j in drop:
                continue

            if corr_matrix.loc[col_i, col_j] > threshold:
                # 保留 IV 更高的，剔除 IV 较低的
                if iv_dict.get(col_i, 0) >= iv_dict.get(col_j, 0):
                    drop.add(col_j)
                    print(f"  [共线性剔除] {col_j:<45s}"
                          f" corr={corr_matrix.loc[col_i, col_j]:.3f}"
                          f" IV={iv_dict.get(col_j, 0):.4f}"
                          f" (保留 {col_i})")
                else:
                    drop.add(col_i)
                    print(f"  [共线性剔除] {col_i:<45s}"
                          f" corr={corr_matrix.loc[col_i, col_j]:.3f}"
                          f" IV={iv_dict.get(col_i, 0):.4f}"
                          f" (保留 {col_j})")

    keep_cols = [c for c in woe_cols if c not in drop]
    print(f"\n[共线性] 原始: {len(woe_cols)} 个 → "
          f"剔除: {len(drop)} 个 → 保留: {len(keep_cols)} 个")

    # 保存保留特征列表
    pd.Series(keep_cols).to_csv(
        os.path.join(FEATURE_DIR, 'lr_features.csv'), index=False
    )
    return keep_cols


# ════════════════════════════════════════════════════════════
#  Part 2：评估指标 + LR 5折CV训练
# ════════════════════════════════════════════════════════════

def calc_ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    计算 KS 统计量（Kolmogorov-Smirnov）。

    KS = max(好人累积占比 - 坏人累积占比)
    衡量模型对好坏人的最大区分能力。

    KS 参考标准（信贷风控行业）：
    - < 0.20 : 模型较弱，需重新审视特征
    - 0.20~0.30 : 可接受（Acceptable）
    - 0.30~0.40 : 良好（Good）
    - > 0.40 : 优秀（Excellent）
    - > 0.75 : 可能过拟合，需排查

    与 AUC 关系：Gini = 2×AUC - 1；KS ≈ Gini（数量级相近但不等价）
    """
    df = pd.DataFrame({'y': y_true, 'p': y_prob})
    df = df.sort_values('p', ascending=False).reset_index(drop=True)

    total_bad  = y_true.sum()
    total_good = len(y_true) - total_bad

    df['cum_bad_rate']  = df['y'].cumsum() / total_bad
    df['cum_good_rate'] = (1 - df['y']).cumsum() / total_good
    df['ks']            = df['cum_bad_rate'] - df['cum_good_rate']

    return df['ks'].max()


def _print_metrics(fold: int, auc: float, ks: float) -> None:
    """单折指标打印，格式统一。"""
    gini = 2 * auc - 1
    print(f"  Fold {fold}  AUC={auc:.4f}  KS={ks:.4f}  Gini={gini:.4f}")


def train_lr_cv(df_woe: pd.DataFrame,
                feature_cols: list) -> tuple:
    """
    Logistic Regression 5折交叉验证训练。

    设计要点：
    1. 使用 StandardScaler 标准化
       - LR 的系数大小与特征量纲直接相关
       - WoE 值通常在 [-3, 3] 之间，量纲已接近，但标准化后系数更稳定
       - 标准化后系数绝对值可直接比较各特征的"影响力大小"

    2. solver='lbfgs'，适合中等规模数据集（<30万样本×50特征）
       - lbfgs：拟牛顿法，收敛快，支持 L2 正则
       - 替代方案：liblinear（小数据），saga（大数据+L1）

    3. class_weight='balanced'
       - 处理1:11的样本不平衡（与LGB的scale_pos_weight等价）
       - balanced 模式：自动将少数类权重乘以 n_samples/(2×n_minority)

    4. C=0.1（正则化强度 = 1/C）
       - C 越小 → 正则化越强 → 系数越收缩 → 防过拟合
       - 评分卡场景偏保守，C=0.1 是常用起点

    参数
    ----
    df_woe       : WoE 编码后的特征矩阵（含 TARGET）
    feature_cols : 共线性处理后保留的 WoE 特征列名列表

    返回
    ----
    oof_pred  : OOF 预测概率数组
    models    : 各折训练好的 (scaler, lr) 元组列表
    cv_metrics: DataFrame，各折 AUC/KS/Gini
    """
    X = df_woe[feature_cols].values
    y = df_woe[TARGET].values

    skf      = StratifiedKFold(n_splits=LGB_CV_FOLDS,
                               shuffle=True, random_state=42)
    oof_pred = np.zeros(len(y))
    models   = []
    rows     = []

    print(f"\n[LR CV] 开始 {LGB_CV_FOLDS} 折交叉验证")
    print(f"  特征数: {len(feature_cols)}  样本量: {len(y)}")

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx],  y[val_idx]

        # 标准化（在训练集上 fit，避免数据泄露）
        scaler  = StandardScaler()
        X_trn_s = scaler.fit_transform(X_trn)
        X_val_s = scaler.transform(X_val)

        lr = LogisticRegression(
            C            = 0.1,
            solver       = 'lbfgs',
            max_iter     = 1000,
            class_weight = 'balanced',
            random_state = 42,
            n_jobs       = -1,
        )
        lr.fit(X_trn_s, y_trn)

        prob = lr.predict_proba(X_val_s)[:, 1]
        oof_pred[val_idx] = prob

        auc = roc_auc_score(y_val, prob)
        ks  = calc_ks(y_val, prob)
        _print_metrics(fold, auc, ks)

        models.append((scaler, lr))
        rows.append({
            'fold': fold,
            'auc' : round(auc, 5),
            'ks'  : round(ks,  5),
            'gini': round(2 * auc - 1, 5),
        })

        # 保存模型
        joblib.dump((scaler, lr),
                    os.path.join(MODEL_DIR, f'lr_fold{fold}.pkl'))

    # OOF 汇总
    oof_auc  = roc_auc_score(y, oof_pred)
    oof_ks   = calc_ks(y, oof_pred)
    oof_gini = 2 * oof_auc - 1

    print(f"\n  ── OOF 汇总 ─────────────────────────────")
    print(f"  OOF AUC  = {oof_auc:.4f}   {'✓ Good' if oof_auc >= 0.80 else '✓ OK' if oof_auc >= 0.75 else '✗ 偏弱'}")
    print(f"  OOF KS   = {oof_ks:.4f}   {'✓ Good' if oof_ks >= 0.40 else '✓ OK' if oof_ks >= 0.30 else '✗ 偏弱'}")
    print(f"  OOF Gini = {oof_gini:.4f}")

    rows.append({
        'fold': 'OOF',
        'auc' : round(oof_auc,  5),
        'ks'  : round(oof_ks,   5),
        'gini': round(oof_gini, 5),
    })
    cv_metrics = pd.DataFrame(rows)
    cv_metrics.to_csv(
        os.path.join(MODEL_DIR, 'lr_cv_metrics.csv'), index=False
    )

    return oof_pred, models, cv_metrics


# ════════════════════════════════════════════════════════════
#  Part 3：系数表 + 方向检验 + 主入口
# ════════════════════════════════════════════════════════════

def build_coef_table(models: list,
                     feature_cols: list,
                     iv_summary: pd.DataFrame) -> pd.DataFrame:
    """
    汇总各折 LR 系数，输出系数稳定性表，并做方向检验。

    评分卡方向检验原则：
    - WoE 与违约概率的关系：WoE 越低（负值）→ 坏人比例越高
    - LR 中：log-odds = β₀ + β₁×WoE₁ + ...
    - WoE 越低 → 预测违约概率越高 → 系数 β 应为负数（负相关）
    - 若某特征系数为正，说明该特征的 WoE 与 log-odds 正相关，
      即 WoE 越高 → 好人越多 → 违约概率越低 → 系数应为负
    - 换言之：所有 WoE 特征的系数理论上应为负数

    注意：WoE 编码后，LR 系数的"业务方向"统一由 WoE 的正负承载，
    LR 系数负号本身就表示"WoE 越高越好"（风险越低），是预期的正确方向。
    若系数为正，说明该特征可能存在 WoE 计算问题或多重共线性残留。

    参数
    ----
    models       : train_lr_cv 返回的 [(scaler, lr), ...] 列表
    feature_cols : WoE 特征列名列表
    iv_summary   : IV 汇总（用于附加 IV 信息）

    返回
    ----
    coef_df : DataFrame，含各折系数、均值、标准差、方向检验结果
    """
    coef_matrix = np.array([lr.coef_[0] for _, lr in models])  # shape=(n_folds, n_feats)

    coef_df = pd.DataFrame({
        'feature'   : feature_cols,
        'coef_mean' : coef_matrix.mean(axis=0).round(6),
        'coef_std'  : coef_matrix.std(axis=0).round(6),
        'coef_min'  : coef_matrix.min(axis=0).round(6),
        'coef_max'  : coef_matrix.max(axis=0).round(6),
    })

    # 各折系数单独列出（便于稳定性观察）
    for i, (_, lr) in enumerate(models, 1):
        coef_df[f'coef_fold{i}'] = lr.coef_[0].round(6)

    # 附加 IV（方便排序参考）
    iv_lookup = {}
    for _, row in iv_summary.iterrows():
        iv_lookup[row['feature'] + '_WOE'] = row['iv']
    coef_df['iv'] = coef_df['feature'].map(iv_lookup).fillna(0.0)

    # ── 方向检验 ──────────────────────────────────────────────
    # 期望：WoE 特征系数为负（WoE 越高 → 越像好人 → 违约概率越低）
    coef_df['direction_ok'] = coef_df['coef_mean'] < 0
    coef_df['direction']    = coef_df['direction_ok'].map(
        {True: '✓ 负（预期）', False: '✗ 正（异常）'}
    )

    # 系数稳定性：cv_std / |cv_mean|，越小越稳定
    coef_df['coef_cv'] = (
        coef_df['coef_std'] / (coef_df['coef_mean'].abs() + 1e-8)
    ).round(4)

    coef_df = coef_df.sort_values('iv', ascending=False).reset_index(drop=True)

    # ── 截距行（供 scorecard.py 使用）────────────────────────
    # LR 截距 β₀ 必须参与评分卡转换，否则各特征分值之和会系统性偏移
    # 将截距存为特殊行，scorecard.py 读取后均摊到每个特征的基础分里
    intercept_vals = np.array([lr.intercept_[0] for _, lr in models])
    intercept_row  = pd.DataFrame([{
        'feature'    : '_INTERCEPT',
        'coef_mean'  : intercept_vals.mean().round(6),
        'coef_std'   : intercept_vals.std().round(6),
        'coef_min'   : intercept_vals.min().round(6),
        'coef_max'   : intercept_vals.max().round(6),
        'iv'         : 0.0,
        'direction_ok': True,
        'direction'  : '(截距)',
        'coef_cv'    : 0.0,
    }])
    coef_df = pd.concat([coef_df, intercept_row], ignore_index=True)

    # ── 方向检验（排除截距行）────────────────────────────────
    abnormal = coef_df[~coef_df['direction_ok'] &
                       (coef_df['feature'] != '_INTERCEPT')]
    if len(abnormal) > 0:
        print(f"\n  [方向检验] ⚠ 发现 {len(abnormal)} 个系数方向异常特征：")
        for _, row in abnormal.iterrows():
            print(f"    {row['feature']:<45s} coef={row['coef_mean']:.4f}"
                  f"  IV={row['iv']:.4f}")
        print("  → 建议检查对应特征的WoE单调性或是否存在共线性残留")
    else:
        print("\n  [方向检验] ✓ 所有特征系数方向正常（均为负）")

    coef_df.to_csv(
        os.path.join(MODEL_DIR, 'lr_coef_table.csv'), index=False
    )
    print(f"  系数表已保存: {os.path.join(MODEL_DIR, 'lr_coef_table.csv')}")

    return coef_df


def run_lr_modeling(df_woe: pd.DataFrame,
                    iv_summary: pd.DataFrame) -> tuple:
    """
    LR 建模完整流程入口，供 main.py Step6 调用。

    流程：
    1. 共线性剔除（corr > 0.85，保留 IV 更高的）
    2. 5折交叉验证训练 LR
    3. 输出系数表 + 方向检验

    参数
    ----
    df_woe     : run_woe_encoding 返回的 WoE 特征矩阵（含 TARGET）
    iv_summary : run_woe_encoding 返回的 IV 汇总

    返回
    ----
    models     : 各折 (scaler, lr) 列表（供 scorecard.py 使用）
    feature_cols : 最终进入模型的特征列名
    coef_df    : 系数稳定性表
    cv_metrics : 各折 AUC/KS/Gini 汇总
    """
    print("\n" + "=" * 55)
    print("[Step 6] Logistic Regression 建模")
    print("=" * 55)

    # Step 6-1: 共线性剔除
    print("\n[Step 6-1] 共线性处理")
    feature_cols = remove_collinear(df_woe, iv_summary)

    if len(feature_cols) == 0:
        raise ValueError("[LR] 共线性剔除后无剩余特征，请检查 WoE 编码结果")

    # Step 6-2: 5折CV训练
    print("\n[Step 6-2] 5折交叉验证")
    oof_pred, models, cv_metrics = train_lr_cv(df_woe, feature_cols)

    # Step 6-3: 系数表 + 方向检验
    print("\n[Step 6-3] 系数表 & 方向检验")
    coef_df = build_coef_table(models, feature_cols, iv_summary)

    print(f"\n  Top-10 特征系数（按IV排序）：")
    display_cols = ['feature', 'coef_mean', 'coef_std', 'iv', 'direction']
    print(coef_df[display_cols].head(10).to_string(index=False))

    print("\n[Step 6] LR 建模完成")
    print("=" * 55)

    return models, feature_cols, coef_df, cv_metrics

