"""
主流程入口
运行方式: python main.py
"""
import pandas as pd
import os
from src.config import DATA_PROCESSED, TARGET_FEATURE_NUM, IMPORTANCE_THRESHOLD
from src.data_loader import load_all
from src.feature_engineering import build_features
from src.feature_selection import run_lgb_cv, select_features, post_lgb_filter
from src.woe_encoding import run_woe_encoding
from src.model_lr import run_lr_modeling
from src.scorecard import run_scorecard


def main():
    # ── Step 1: 读取数据 ──────────────────────────────────
    print("\n>>> Step 1: 读取数据")
    data = load_all()

    # ── Step 2: 特征工程 ──────────────────────────────────
    print("\n>>> Step 2: 特征工程")
    processed_path = os.path.join(DATA_PROCESSED, 'features.pkl')

    if os.path.exists(processed_path):
        print("  检测到缓存文件，直接加载...")
        df = pd.read_pickle(processed_path)
    else:
        df = build_features(data)
        df.to_pickle(processed_path)
        print(f"  特征表已缓存: {processed_path}")

    print(f"  特征表 shape: {df.shape}")

    # ── Step 3: LGB CV + 特征重要性 ───────────────────────
    print("\n>>> Step 3: LGB 交叉验证 + 特征重要性")
    imp_cache = os.path.join('outputs', 'features', 'lgb_importance.csv')
    if os.path.exists(imp_cache):
        print("  检测到 LGB importance 缓存，直接加载（跳过重新训练）...")
        imp_df   = pd.read_csv(imp_cache)
        oof_pred = None   # 缓存模式下无 OOF，后续不使用
    else:
        oof_pred, imp_df = run_lgb_cv(df)

    print("\n  Top 20 重要特征:")
    print(imp_df.head(20).to_string(index=False))

    # ── Step 4: LGB Top-N 筛选 ───────────────────────────
    print("\n>>> Step 4: 特征筛选")
    selected_features = select_features(
        imp_df, df, top_n=TARGET_FEATURE_NUM
    )
    print(f"  LGB Top-{TARGET_FEATURE_NUM} 特征数: {len(selected_features)}")

    # ── Step 4.5: 三层后处理精筛（Layer1 + Layer3）─────────
    # Layer2（相关系数+IV）在 WoE 拟合后执行，此处先做 Layer1 + Layer3
    # 说明：Layer2 需要 IV 值，WoE 尚未运行，故分两阶段执行
    print("\n>>> Step 4.5: 后处理精筛（Layer1 规则 + Layer3 EXT_SOURCE 业务过滤）")
    selected_features = post_lgb_filter(
        selected_features, df, iv_summary=None  # iv_summary=None → 跳过 Layer2
    )
    print(f"  精筛后特征数: {len(selected_features)}")

    # ── Step 5: WoE 编码 ──────────────────────────────────
    print("\n>>> Step 5: WoE 编码")
    df_woe, woe_map, iv_summary = run_woe_encoding(df, selected_features)

    # ── Step 5.5: Layer2 相关系数+IV 过滤（需要 WoE IV 结果）──
    # 现在 iv_summary 已经有了，可以执行数据驱动的相关系数过滤
    # 注意：此步只过滤 selected_features 列表，不改变已编码的 df_woe
    # df_woe 中多余的 _WOE 列不影响 remove_collinear（model_lr 内部还会再做一次）
    print("\n>>> Step 5.5: Layer2 相关系数过滤（corr > 0.85，保留 IV 高者）")
    selected_features = post_lgb_filter(
        selected_features, df,
        iv_summary   = iv_summary,
        corr_threshold = 0.85
    )
    # 同步更新 iv_summary，只保留精筛后的特征行
    iv_summary = iv_summary[iv_summary['feature'].isin(selected_features)].copy()
    # 同步更新 df_woe，只保留精筛后特征的 _WOE 列
    keep_woe_cols = ['SK_ID_CURR', 'TARGET'] + \
                    [f + '_WOE' for f in selected_features
                     if f + '_WOE' in df_woe.columns]
    df_woe = df_woe[[c for c in keep_woe_cols if c in df_woe.columns]].copy()
    print(f"  Layer2 过滤后特征数: {len(selected_features)}")

    print("\n  IV 汇总 Top-20:")
    print(iv_summary.head(20).to_string(index=False))

    # ── Step 6: LR 建模 ──────────────────────────────────
    print("\n>>> Step 6: Logistic Regression 建模")
    models, lr_features, coef_df, cv_metrics = run_lr_modeling(
        df_woe, iv_summary
    )

    # ── Step 7: 评分卡转换 ────────────────────────────────────
    print("\n>>> Step 7: 评分卡生成")
    scorecard = run_scorecard(
        df_raw     = df,          # 原始特征表（含TARGET），用于分布校验
        woe_map    = woe_map,
        coef_df    = coef_df,
        iv_summary = iv_summary,
        validate   = True,        # 设为 False 可跳过批量评分，节省时间
    )
    print("\n>>> 全流程完成")


if __name__ == '__main__':
    main()