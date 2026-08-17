"""
Case Study 深度分析
  Part A: 典型案例解剖（可解释性）
  Part B: 错误案例分析（边界理解）
"""
import pandas as pd
import numpy as np
import pickle, sys, os
sys.path.insert(0, '.')

from src.scorecard import score_single, _find_bin

# ─────────────────────────────────────────────────────
# 加载工件
# ─────────────────────────────────────────────────────
sc_df  = pd.read_csv('outputs/scorecard/scorecard.csv')
woe_df = pd.read_csv('outputs/features/woe_map.csv')
dist   = pd.read_csv('outputs/scorecard/score_dist.csv')
dist.columns = ['score', 'target', 'bin']

# 重建 woe_map dict（{feature: DataFrame}）
woe_map = {feat: grp.reset_index(drop=True)
           for feat, grp in woe_df.groupby('feature')}

# 加载原始特征表（仅取前 5000 行，与校验采样一致）
df_raw = pd.read_pickle('data/processed/features.pkl').head(5000)
df_raw = df_raw.reset_index(drop=True)
dist   = dist.reset_index(drop=True)
df_raw['__score__'] = dist['score']
df_raw['TARGET']    = dist['target']

# ─────────────────────────────────────────────────────
# 工具：打印单人评分明细
# ─────────────────────────────────────────────────────
def print_score_detail(row: pd.Series, label: str = ""):
    applicant = row.to_dict()
    result = score_single(applicant, sc_df, woe_map)
    total  = result['total_score']
    actual = int(row.get('TARGET', -1))
    act_str = '坏客户(违约)' if actual == 1 else '好客户(正常)'

    print(f"\n{'='*64}")
    print(f"  {label}")
    print(f"  实际标签  : {act_str}   |   模型评分 : {total:.1f} 分")
    print(f"{'='*64}")
    print(f"  {'特征':<30} {'原始值':>12}  {'分箱':>30}  {'得分':>6}")
    print(f"  {'-'*83}")

    details = sorted(result['details'], key=lambda x: -abs(
        sc_df[(sc_df.feature == x['feature']) &
              (sc_df.bin_label == x['bin_label'])]['score'].values[0]
        - sc_df[sc_df.feature == x['feature']]['score'].mean()
        if len(sc_df[(sc_df.feature == x['feature']) &
                     (sc_df.bin_label == x['bin_label'])]) > 0 else 0
    ))

    for d in result['details']:
        feat_scores = sc_df[sc_df.feature == d['feature']]['score']
        feat_mean   = feat_scores.mean() if len(feat_scores) > 0 else 0
        delta       = d['score'] - feat_mean
        arrow       = f"+{delta:+.1f}" if delta >= 0 else f"{delta:+.1f}"
        rv = d['raw_value']
        rv_str = f"{rv:.4f}" if isinstance(rv, float) else str(rv)
        print(f"  {d['feature']:<30} {rv_str:>12}  {d['bin_label']:>30}  "
              f"{d['score']:>6.1f}  ({arrow})")

    print(f"  {'─'*83}")
    print(f"  {'总分':>67}  {total:>6.1f}")
    return result


# ─────────────────────────────────────────────────────
# Part A: 典型案例解剖
# ─────────────────────────────────────────────────────
print("\n" + "▓"*68)
print("  PART A：典型案例解剖 — 证明模型「可解释」")
print("▓"*68)

# A1: 典型好客户（高分 + 实际好）
good_pool = df_raw[(df_raw.TARGET == 0) & (df_raw['__score__'] >= 748)].sample(
    5, random_state=42)
case_good = good_pool.iloc[0]
r_good = print_score_detail(case_good, "A1：典型优质客户（高分好人）— 解释为什么模型信任他")

# A2: 典型坏客户（低分 + 实际坏）
bad_pool = df_raw[(df_raw.TARGET == 1) & (df_raw['__score__'] <= 706)].copy()
bad_pool = bad_pool.sort_values('__score__')
case_bad = bad_pool.iloc[0]
r_bad = print_score_detail(case_bad, "A2：典型高危客户（低分坏人）— 解释为什么模型拒绝他")

# A3: 中等风险客户（灰度区）
mid_pool = df_raw[(df_raw.TARGET == 0) &
                  (df_raw['__score__'] >= 718) &
                  (df_raw['__score__'] <= 725)].sample(3, random_state=7)
case_mid = mid_pool.iloc[0]
r_mid = print_score_detail(case_mid, "A3：灰度区客户（中等分好人）— 解释为什么需要人工复核")

# Part A 小结
print("\n\n" + "─"*68)
print("  Part A 可解释性小结")
print("─"*68)
# 提取三个客户的 EXT_SOURCE_MEAN 值对比
def get_feat(row, feat):
    return row.get(feat, np.nan)

print(f"\n  关键特征对比（三类客户）:")
feats_compare = ['EXT_SOURCE_MEAN', 'EXT_SOURCE_2_3', 'EXT_SOURCE_1_2',
                 'EMPLOYED_EXT2_CROSS', 'CREDIT_GOODS_RATIO',
                 'BUREAU_UTIL_MAX', 'INST_PAYMENT_RATIO_MEAN']
print(f"  {'特征':<30} {'A1优质':>10} {'A2高危':>10} {'A3灰度':>10}")
print(f"  {'─'*63}")
for feat in feats_compare:
    v1 = case_good.get(feat, np.nan)
    v2 = case_bad.get(feat, np.nan)
    v3 = case_mid.get(feat, np.nan)
    fmt = lambda x: f"{x:.4f}" if isinstance(x, float) and not np.isnan(x) else "NaN"
    print(f"  {feat:<30} {fmt(v1):>10} {fmt(v2):>10} {fmt(v3):>10}")

# ─────────────────────────────────────────────────────
# Part B: 错误案例分析
# ─────────────────────────────────────────────────────
print("\n\n" + "▓"*68)
print("  PART B：错误案例分析 — 证明「理解模型边界」")
print("▓"*68)

# B1: 假阳性（False Positive）—— 模型评高分，实际违约
fp_pool = df_raw[(df_raw.TARGET == 1) & (df_raw['__score__'] >= 730)].copy()
print(f"\n  [FP池] 高分坏客户（分≥730且实际违约）: {len(fp_pool)} 人")
if len(fp_pool) >= 2:
    fp_pool = fp_pool.sort_values('__score__', ascending=False)
    case_fp = fp_pool.iloc[0]
    r_fp = print_score_detail(case_fp, "B1：假阳性 — 模型评高分但实际违约（漏网之鱼）")

# B2: 假阴性（False Negative）—— 模型评低分，实际正常
fn_pool = df_raw[(df_raw.TARGET == 0) & (df_raw['__score__'] <= 710)].copy()
print(f"\n  [FN池] 低分好客户（分≤710且实际正常）: {len(fn_pool)} 人")
if len(fn_pool) >= 2:
    fn_pool = fn_pool.sort_values('__score__')
    case_fn = fn_pool.iloc[0]
    r_fn = print_score_detail(case_fn, "B2：假阴性 — 模型评低分但实际正常（误杀好人）")

# B3：FP / FN 特征分布对比
print("\n\n" + "─"*68)
print("  Part B 错误模式诊断")
print("─"*68)

# FP特征均值 vs 真正好客户均值（分≥730，TARGET=0）
tp_pool = df_raw[(df_raw.TARGET == 0) & (df_raw['__score__'] >= 730)]

diag_feats = ['EXT_SOURCE_MEAN', 'EXT_SOURCE_2_3', 'EXT_SOURCE_1_2',
              'EMPLOYED_EXT2_CROSS', 'CREDIT_GOODS_RATIO',
              'BUREAU_UTIL_MAX', 'INST_PAYMENT_RATIO_MEAN',
              'DAYS_BIRTH', 'BUREAU_DAYS_CREDIT_MEAN']

# 过滤缺失值占位符 -999（建模时填充值，不参与统计比较）
def clean_feat(series):
    return series.replace(-999.0, np.nan).replace(-999, np.nan)

print(f"\n  高分区（≥730）好/坏客户特征均值对比:")
print(f"  {'特征':<30} {'真阳(TP)均值':>14} {'假阳(FP)均值':>14} {'差异%':>8}")
print(f"  {'─'*70}")
for feat in diag_feats:
    if feat not in df_raw.columns:
        continue
    tp_mean = clean_feat(tp_pool[feat]).mean()
    fp_mean = clean_feat(fp_pool[feat]).mean() if len(fp_pool) > 0 else np.nan
    if not np.isnan(tp_mean) and not np.isnan(fp_mean) and tp_mean != 0:
        diff_pct = (fp_mean - tp_mean) / abs(tp_mean) * 100
        flag = " ★" if abs(diff_pct) > 15 else ""
        print(f"  {feat:<30} {tp_mean:>14.4f} {fp_mean:>14.4f} {diff_pct:>7.1f}%{flag}")

# FN特征均值 vs 真负好客户均值（分≤710，TARGET=1）
tn_pool = df_raw[(df_raw.TARGET == 1) & (df_raw['__score__'] <= 710)]

print(f"\n  低分区（≤710）坏/好客户特征均值对比:")
print(f"  {'特征':<30} {'真阴(TN)均值':>14} {'假阴(FN)均值':>14} {'差异%':>8}")
print(f"  {'─'*70}")
for feat in diag_feats:
    if feat not in df_raw.columns:
        continue
    tn_mean = clean_feat(tn_pool[feat]).mean()
    fn_mean = clean_feat(fn_pool[feat]).mean() if len(fn_pool) > 0 else np.nan
    if not np.isnan(tn_mean) and not np.isnan(fn_mean) and tn_mean != 0:
        diff_pct = (fn_mean - tn_mean) / abs(tn_mean) * 100
        flag = " ★" if abs(diff_pct) > 15 else ""
        print(f"  {feat:<30} {tn_mean:>14.4f} {fn_mean:>14.4f} {diff_pct:>7.1f}%{flag}")

# B4: 按分段统计误判率
print("\n\n  各分段混淆矩阵（误判率分析）:")
print(f"  {'分段':>26} {'总人数':>7} {'坏人数':>7} {'实际坏率':>9} "
      f"{'FP(高分坏)':>10} {'FN(低分好)':>10}")
print(f"  {'─'*75}")
bins  = dist.copy() if 'bin' in dist.columns else None
df_raw['__bin__'] = pd.cut(df_raw['__score__'],
    bins=[692, 706, 714, 720, 725, 730, 734, 738, 742, 748, 764],
    labels=['692~706','706~714','714~720','720~725','725~730',
            '730~734','734~738','738~742','742~748','748~763'])

for seg, g in df_raw.groupby('__bin__', observed=True):
    n_total = len(g)
    n_bad   = g['TARGET'].sum()
    bad_r   = n_bad / n_total if n_total > 0 else 0
    fp_n    = g[g['TARGET'] == 1]   # 该分段中坏客户（误为高分/低分时是FP或FN）
    fn_n    = g[g['TARGET'] == 0]
    print(f"  {str(seg):>26} {n_total:>7,} {int(n_bad):>7,} {bad_r:>9.2%} "
          f"{len(fp_n):>10,} {len(fn_n):>10,}")

print("\n\n" + "─"*68)
print("  Part B 结论：模型边界与失效原因")
print("─"*68)
print("""
  1. 假阳性（漏网）的根本原因：
     → 欺诈型违约者会"伪造"良好的外部征信（EXT_SOURCE 高）
     → 但还款行为数据（INST_PAYMENT_RATIO_MEAN、近6M行为）
       往往还来不及恶化，模型无法提前识别
     → 对策：引入申请行为特征（多头查询次数）、反欺诈规则并联

  2. 假阴性（误杀）的根本原因：
     → 外部征信低不代表近期会违约
     → 部分人是"历史信用薄弱但当前稳定就业"，
       EMPLOYED_EXT2_CROSS 已试图刻画但权重不足
     → 对策：灰度区人工复核时优先看就业稳定性 + 近期还款行为

  3. 关键盲区（模型看不到的信号）：
     → 宏观因素（失业率、利率变动）：静态评分卡不感知
     → 家庭/社会层面压力：CNT_FAM_MEMBERS 等变量IV太低未纳入
     → 欺诈行为：需要单独的欺诈模型并联使用
""")
