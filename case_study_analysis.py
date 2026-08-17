import pandas as pd
import numpy as np

dist = pd.read_csv('outputs/scorecard/score_dist.csv')
dist.columns = ['score', 'target', 'bin']

n_total = len(dist)
n_bad   = int(dist.target.sum())
n_good  = n_total - n_bad

print(f"样本总量  : {n_total:,}")
print(f"坏客户数  : {n_bad:,}  ({n_bad/n_total:.2%})")
print(f"好客户数  : {n_good:,}")
print(f"\n评分范围  : {dist.score.min():.1f} ~ {dist.score.max():.1f}")
print(f"中位数    : {dist.score.median():.1f}")
print(f"均值      : {dist.score.mean():.1f}")
print(f"好客户均分: {dist[dist.target==0].score.mean():.1f}")
print(f"坏客户均分: {dist[dist.target==1].score.mean():.1f}")
print(f"好坏均分差: {dist[dist.target==0].score.mean() - dist[dist.target==1].score.mean():.1f} 分")

grp = dist.groupby('bin').agg(
    total     = ('target', 'count'),
    bad       = ('target', 'sum'),
    avg_score = ('score',  'mean'),
).reset_index()
grp['good']     = grp['total'] - grp['bad']
grp['bad_rate'] = grp['bad'] / grp['total']
grp = grp.sort_values('avg_score').reset_index(drop=True)

grp['cum_bad']   = grp['bad'].cumsum()
grp['cum_good']  = grp['good'].cumsum()
grp['cum_bad_pct']  = grp['cum_bad']  / n_bad
grp['cum_good_pct'] = grp['cum_good'] / n_good
grp['cum_pass_rate'] = (grp['cum_good'] + grp['cum_bad']) / n_total  # 整体累积通过
grp['KS'] = (grp['cum_bad_pct'] - grp['cum_good_pct']).abs()

print("\n\n===== 分箱坏率分布（低分→高分，拒绝→通过方向）=====")
header = f"{'分箱区间':<26} {'均分':>6} {'总量':>7} {'坏客户':>7} {'坏率':>7} | {'累积通过%':>9} {'累积坏客捕获':>12}"
print(header)
print("-" * len(header))
for _, r in grp.iterrows():
    print(f"{r['bin']:<26} {r['avg_score']:>6.1f} {r['total']:>7,} {r['bad']:>7,} "
          f"{r['bad_rate']:>7.2%} | {r['cum_pass_rate']:>9.2%} {r['cum_bad_pct']:>12.2%}")

print("\n\n===== KS 拆解表 =====")
print(f"{'分箱区间':<26} {'累积坏客TPR':>11} {'累积好客FPR':>11} {'KS':>7}")
print("-" * 60)
for _, r in grp.iterrows():
    flag = " ← KS MAX" if abs(r['KS'] - grp['KS'].max()) < 1e-6 else ""
    print(f"{r['bin']:<26} {r['cum_bad_pct']:>11.3f} {r['cum_good_pct']:>11.3f} {r['KS']:>7.3f}{flag}")

print(f"\n模型 KS = {grp['KS'].max():.4f}")

print("\n\n===== 候选切分点（通过率目标）=====")
print(f"{'通过率目标':<12} {'最低分档':<26} {'均分':>6} {'该档坏率':>10} {'以下坏率':>10} {'捕获坏客占比':>12}")
print("-" * 85)
for pass_rate in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    candidates = grp[grp['cum_pass_rate'] <= pass_rate]
    if candidates.empty:
        continue
    cut = candidates.iloc[-1]
    below_bad_rate = cut['cum_bad'] / (cut['cum_bad'] + cut['cum_good']) if (cut['cum_bad'] + cut['cum_good']) > 0 else 0
    print(f"  pass≤{pass_rate:.0%}   {cut['bin']:<26} {cut['avg_score']:>6.1f} "
          f"{cut['bad_rate']:>10.2%} {below_bad_rate:>10.2%} {cut['cum_bad_pct']:>12.1%}")

print("\n\n===== 三档策略切分建议 =====")
# 拒绝区（高风险）: 坏率 > 10%
reject = grp[grp['bad_rate'] > 0.10].tail(1)
# 灰度区（中风险）: 坏率 5%~10%
grey   = grp[(grp['bad_rate'] >= 0.05) & (grp['bad_rate'] <= 0.10)]
# 通过区（低风险）: 坏率 < 5%
accept = grp[grp['bad_rate'] < 0.05]

if not reject.empty:
    r = reject.iloc[0]
    print(f"  拒绝区上沿（分<{r['avg_score']:.0f}）: 坏率={r['bad_rate']:.2%}, 人数占比={r['total']/n_total:.1%}")
if not grey.empty:
    low_s  = grey.iloc[0]['avg_score']
    high_s = grey.iloc[-1]['avg_score']
    grey_bad = grey['bad'].sum() / grey['total'].sum()
    print(f"  灰度区（{low_s:.0f}~{high_s:.0f}分）: 平均坏率={grey_bad:.2%}, "
          f"人数占比={grey['total'].sum()/n_total:.1%}")
if not accept.empty:
    a = accept.iloc[0]
    print(f"  通过区下沿（分≥{a['avg_score']:.0f}）: 坏率={a['bad_rate']:.2%}, "
          f"人数占比={accept['total'].sum()/n_total:.1%}")
