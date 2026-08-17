import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.config import (LGB_PARAMS, LGB_CV_FOLDS, LGB_EARLY_STOP,
                        TARGET, ID_COL, IMPORTANCE_THRESHOLD,
                        MODEL_DIR, FEATURE_DIR)


def run_lgb_cv(df: pd.DataFrame):
    """
    LGB K折交叉验证
    作用：评估特征体系质量 + 获取特征重要性
    返回：oof预测值, 特征重要性DataFrame
    """
    drop_cols = [ID_COL, TARGET]
    feat_cols  = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].values
    y = df[TARGET].values

    skf      = StratifiedKFold(n_splits=LGB_CV_FOLDS,
                               shuffle=True, random_state=42)
    oof_pred = np.zeros(len(y))
    imp_list = []

    print(f"\n[LGB CV] 开始 {LGB_CV_FOLDS} 折交叉验证")
    print(f"  特征数量: {len(feat_cols)}")
    print(f"  样本量:   {len(y)}")

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_trn, y_trn = X[trn_idx], y[trn_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_trn, y_trn,
            eval_set        = [(X_val, y_val)],
            callbacks       = [
                lgb.early_stopping(LGB_EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=200)
            ]
        )

        oof_pred[val_idx] = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, oof_pred[val_idx])
        print(f"  Fold {fold} AUC: {auc:.5f}  "
              f"best_iter: {model.best_iteration_}")

        # 记录本折特征重要性
        imp_list.append(pd.DataFrame({
            'feature'   : feat_cols,
            'importance': model.feature_importances_,
            'fold'      : fold,
        }))

        # 保存模型
        joblib.dump(model, os.path.join(MODEL_DIR, f'lgb_fold{fold}.pkl'))

    # 汇总
    oof_auc = roc_auc_score(y, oof_pred)
    print(f"\n  OOF AUC (全量): {oof_auc:.5f}")

    imp_df = (pd.concat(imp_list)
                .groupby('feature')['importance']
                .mean()
                .reset_index()
                .sort_values('importance', ascending=False))

    imp_df.to_csv(
        os.path.join(FEATURE_DIR, 'lgb_importance.csv'), index=False
    )
    print(f"  特征重要性已保存")

    return oof_pred, imp_df


def select_features(imp_df: pd.DataFrame,
                    df: pd.DataFrame,
                    top_n: int = None,
                    threshold: float = None) -> list:
    """
    根据LGB重要性筛选特征
    优先使用 top_n，其次使用 importance threshold
    """
    if top_n is not None:
        selected = imp_df.head(top_n)['feature'].tolist()
        print(f"\n[特征筛选] Top-{top_n} 特征: {len(selected)} 个")
    elif threshold is not None:
        selected = imp_df[
            imp_df['importance'] >= threshold
        ]['feature'].tolist()
        print(f"\n[特征筛选] importance >= {threshold}: {len(selected)} 个")
    else:
        raise ValueError("top_n 和 threshold 至少指定一个")

    # 保存筛选后特征列表
    pd.Series(selected).to_csv(
        os.path.join(FEATURE_DIR, 'selected_features.csv'), index=False
    )
    return selected


def post_lgb_filter(selected: list,
                    df: pd.DataFrame,
                    iv_summary: pd.DataFrame = None,
                    corr_threshold: float = 0.85) -> list:
    """
    LGB Top-N 之后的三层精筛，在进入 WoE 编码之前执行。

    三层过滤策略
    ────────────────────────────────────────────────────────
    Layer 1 ─ 规则过滤（确定性冗余）
      · 完全线性相关的特征对（|corr|≥0.99）只保留一个
        规则：优先保留名称更短的原始值，或由调用方通过 FORCE_DROP 指定
      · 由配置项 LAYER1_DROP 硬编码已知冗余特征列表

    Layer 2 ─ 相关系数过滤（数据驱动，阈值0.85）
      · 计算 Pearson 相关矩阵
      · |corr| > corr_threshold 的特征对，删 IV 更低的那个
      · IV 来源：iv_summary 参数（若为 None 则跳过此层）

    Layer 3 ─ 业务逻辑过滤（可解释性，硬编码规则）
      · EXT_SOURCE 组：保留最具代表性的子集，删除与保留项高度共线的冗余
        依据：相关矩阵分析（EXT3≈2_3, EXT1≈1_2, EXT2≈2_AGE）
      · 保留信息量更高的交叉特征，删除低IV的原始值

    参数
    ────
    selected       : select_features() 返回的 Top-N 特征列表
    df             : 特征宽表（用于计算相关矩阵）
    iv_summary     : fit_woe() 返回的 IV 汇总（Layer2 使用；若已有可传入）
    corr_threshold : Layer2 相关系数阈值，默认 0.85

    返回
    ────
    filtered : 精筛后的特征列表（去重、有序）
    """
    print("\n" + "=" * 55)
    print("[后处理精筛] 三层过滤")
    print("=" * 55)
    print(f"  输入特征数: {len(selected)}")

    current = [f for f in selected if f in df.columns]

    # ════════════════════════════════════════════════════
    # Layer 1：规则过滤（硬编码已知冗余）
    # ════════════════════════════════════════════════════
    # 依据：
    #   · AGE_YEARS = -DAYS_BIRTH/365，corr=-1.0，完全线性相关
    #     → 保留 DAYS_BIRTH（原始值，WoE 切分更直接）
    #   · 原始金额与衍生比率同时存在时，保留比率（信息更独立）
    #     AMT_CREDIT 与 AMT_GOODS_PRICE corr=0.987，且比率 CREDIT_GOODS_RATIO
    #     已充分捕捉两者之差，AMT_CREDIT 可删
    LAYER1_DROP = [
        'AGE_YEARS',       # 完全等于 -DAYS_BIRTH/365，corr=-1.0
        'AMT_CREDIT',      # corr=0.987 与 AMT_GOODS_PRICE；CREDIT_GOODS_RATIO 已覆盖套现信号
        'AMT_GOODS_PRICE', # 与 CREDIT_GOODS_RATIO 共线；系数方向异常(+)、IV=0.043(弱)
    ]
    l1_dropped = [f for f in LAYER1_DROP if f in current]
    current    = [f for f in current if f not in LAYER1_DROP]
    print(f"\n[Layer1 规则过滤] 删除 {len(l1_dropped)} 个:")
    for f in l1_dropped:
        print(f"  ✗ {f}")

    # ════════════════════════════════════════════════════
    # Layer 2：相关系数过滤（数据驱动）
    # ════════════════════════════════════════════════════
    if iv_summary is not None:
        # 构建 IV 查找字典
        iv_dict = dict(zip(iv_summary['feature'], iv_summary['iv']))

        avail_in_df = [f for f in current if f in df.columns]
        corr_mat    = df[avail_in_df].corr().abs()

        l2_dropped  = set()
        for i in range(len(avail_in_df)):
            for j in range(i + 1, len(avail_in_df)):
                fi, fj = avail_in_df[i], avail_in_df[j]
                if fi in l2_dropped or fj in l2_dropped:
                    continue
                if corr_mat.loc[fi, fj] > corr_threshold:
                    iv_i = iv_dict.get(fi, 0.0)
                    iv_j = iv_dict.get(fj, 0.0)
                    drop_f  = fj if iv_i >= iv_j else fi
                    keep_f  = fi if iv_i >= iv_j else fj
                    l2_dropped.add(drop_f)
                    print(f"  [Layer2] ✗ {drop_f:<35s} corr={corr_mat.loc[fi,fj]:.3f}"
                          f"  IV={iv_dict.get(drop_f,0):.4f}  (保留 {keep_f})")

        current = [f for f in current if f not in l2_dropped]
        print(f"[Layer2 相关系数过滤] 删除 {len(l2_dropped)} 个")
    else:
        print("[Layer2] 未传入 iv_summary，跳过相关系数过滤")

    # ════════════════════════════════════════════════════
    # Layer 3：EXT_SOURCE 业务逻辑过滤
    # ════════════════════════════════════════════════════
    # 相关矩阵分析结论（数据驱动 + 业务逻辑双重验证）：
    #   EXT_SOURCE_3   vs EXT_SOURCE_2_3:  corr=0.996 → 保留 2_3（IV=0.395 > 0.264）
    #   EXT_SOURCE_1   vs EXT_SOURCE_1_2:  corr=0.999 → 保留 1_2（IV=0.152 > 0.105）
    #   EXT_SOURCE_2   vs EXT_SOURCE_2_AGE:corr=0.975 → 删除两者（都删）
    #   EXT_SOURCE_2_AGE: 删 EXT_SOURCE_2 后仍与 MEAN/2_3/1_2 共线，系数方向异常(+0.035)
    #     根因：EXT2_AGE = EXT2 × AGE，与含 EXT2 信息的 MEAN/2_3/1_2 情报残留重叠
    #     信号已被 EXT_SOURCE_MEAN（IV=0.607）充分贡献，删除不损失信息
    #   EXT_SOURCE_STD: IV=0.018，无预测力（此处作为预防性删除）
    #
    # 最终 EXT_SOURCE 保留集：MEAN, 2_3, 1_2（共 3 个，覆盖 3 个独立维度）
    LAYER3_EXT_DROP = [
        'EXT_SOURCE_3',    # corr=0.996 与 EXT_SOURCE_2_3
        'EXT_SOURCE_1',    # corr=0.999 与 EXT_SOURCE_1_2
        'EXT_SOURCE_2',    # corr=0.975 与 EXT_SOURCE_2_AGE（两者同时删）
        'EXT_SOURCE_2_AGE',# 共线残留，系数方向异常(+)；信号已被 MEAN/2_3/1_2 覆盖
        'EXT_SOURCE_STD',  # IV=0.018，无预测力
    ]
    l3_dropped = [f for f in LAYER3_EXT_DROP if f in current]
    current    = [f for f in current if f not in LAYER3_EXT_DROP]
    print(f"\n[Layer3 EXT_SOURCE 业务过滤] 删除 {len(l3_dropped)} 个:")
    for f in l3_dropped:
        print(f"  ✗ {f}")

    print(f"\n[后处理精筛] 完成: {len(selected)} → {len(current)} 个特征")
    print(f"  保留特征列表:")
    for i, f in enumerate(current, 1):
        print(f"    {i:2d}. {f}")

    # 保存精筛后特征列表
    pd.Series(current).to_csv(
        os.path.join(FEATURE_DIR, 'selected_features_filtered.csv'), index=False
    )
    return current