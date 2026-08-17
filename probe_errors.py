import pandas as pd, numpy as np, sys
sys.path.insert(0, '.')

df = pd.read_pickle('data/processed/features.pkl').head(5000).reset_index(drop=True)
dist = pd.read_csv('outputs/scorecard/score_dist.csv')
dist.columns = ['score', 'target', 'bin']
df['__score__'] = dist['score']
df['TARGET'] = dist['target']

fp = df[(df.TARGET == 1) & (df['__score__'] >= 730)].sort_values('__score__', ascending=False)
fn = df[(df.TARGET == 0) & (df['__score__'] <= 710)].sort_values('__score__')

key_feats = [
    'EXT_SOURCE_MEAN', 'EXT_SOURCE_2_3', 'EMPLOYED_EXT2_CROSS',
    'INST_PAYMENT_RATIO_MEAN', 'INST_RECENT_PAY_RATIO_6M',
    'BUREAU_UTIL_MAX', 'BUREAU_DAYS_CREDIT_MEAN',
    'CREDIT_GOODS_RATIO', 'DAYS_BIRTH', 'CREDIT_TERM',
    'PREV_REFUSED_RATE', 'BUREAU_BB_DPD_RATE_MEAN',
]

def fmt(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or v == -999.0)):
        return 'NaN'
    return f'{float(v):.3f}'

print('=== FP池前5个（高分坏客户）===')
for i, (idx, row) in enumerate(fp.head(5).iterrows()):
    print(f'FP#{i+1}  score={row["__score__"]:.1f}')
    for f in key_feats:
        v = row.get(f, np.nan)
        if isinstance(v, float) and v == -999.0:
            v = np.nan
        print(f'  {f:<35} {fmt(v)}')
    print()

print()
print('=== FN池前5个（低分好客户）===')
for i, (idx, row) in enumerate(fn.head(5).iterrows()):
    print(f'FN#{i+1}  score={row["__score__"]:.1f}')
    for f in key_feats:
        v = row.get(f, np.nan)
        if isinstance(v, float) and v == -999.0:
            v = np.nan
        print(f'  {f:<35} {fmt(v)}')
    print()
