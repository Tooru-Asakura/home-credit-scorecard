"""
WoE 编码模块
流程：分箱 → 计算WoE/IV → 单调性处理 → IV筛选 → 转换
"""
import numpy as np
import pandas as pd
import os
from src.config import FEATURE_DIR, SCORECARD_DIR, TARGET

# ── WoE/IV 筛选阈值 ────────────────────────────────────────
IV_THRESHOLD   = 0.02   # 低于此值视为无预测力，剔除
MAX_BINS       = 10     # 数值型最大分箱数（等频）
MIN_BIN_RATE   = 0.05   # 每个分箱最少占总样本的比例（防止过小箱）


# ════════════════════════════════════════════════════════════
#  Part 1：分箱函数
# ════════════════════════════════════════════════════════════

def bin_numeric(x: pd.Series, y: pd.Series,
                max_bins: int = MAX_BINS,
                min_bin_rate: float = MIN_BIN_RATE) -> np.ndarray:
    """
    数值型特征等频分箱，返回分箱边界数组。

    原理：
    - 等频分箱（每箱样本数相近），保证每个箱有足够统计量计算WoE
    - 合并过小的箱（< min_bin_rate），防止出现0好/0坏导致WoE无穷大
    - 合并纯好人箱或纯坏人箱（WoE计算需要每箱都有好坏样本）

    参数
    ----
    x            : 特征列（已对齐索引）
    y            : TARGET列（0=好人，1=坏人）
    max_bins     : 最大分箱数
    min_bin_rate : 单箱最小样本占比

    返回
    ----
    bins : np.ndarray，分箱边界（含 -inf/+inf）
    """
    # 过滤缺失值，只用非空值确定边界
    mask   = x.notna()
    x_valid = x[mask]
    y_valid = y[mask]

    if len(x_valid) == 0:
        return np.array([-np.inf, np.inf])

    # 等频分箱：按分位数切割
    quantiles = np.linspace(0, 100, max_bins + 1)
    raw_bins  = np.unique(np.percentile(x_valid, quantiles))

    # 保证首尾为 -inf/+inf
    raw_bins[0]  = -np.inf
    raw_bins[-1] = np.inf

    bins = raw_bins.copy()

    # 迭代合并：处理过小箱 or 纯好/纯坏箱
    # 最多循环 max_bins 次，防止死循环
    for _ in range(max_bins):
        if len(bins) <= 2:   # 只剩一个箱，退出
            break

        # 按当前 bins 切割，统计每箱的好坏人数
        labels  = pd.cut(x_valid, bins=bins, include_lowest=True)
        summary = (pd.DataFrame({'bin': labels, 'y': y_valid})
                   .groupby('bin', observed=True)['y']
                   .agg(['count', 'sum']))
        summary['bad']  = summary['sum']
        summary['good'] = summary['count'] - summary['sum']
        summary['rate'] = summary['count'] / len(x_valid)

        # 找需要合并的箱：过小 OR 纯好人 OR 纯坏人
        need_merge = (
            (summary['rate']  < min_bin_rate) |
            (summary['bad']  == 0) |
            (summary['good'] == 0)
        )

        if not need_merge.any():
            break   # 所有箱都合格，退出

        # 找第一个需要合并的箱的索引，与相邻箱合并（删除右边界）
        merge_idx = need_merge.values.argmax()
        # 删除对应右边界（bins[merge_idx+1]），实现与右侧箱合并
        # 若已是最后一箱，则与左侧合并（删除左边界）
        if merge_idx < len(bins) - 2:
            bins = np.delete(bins, merge_idx + 1)
        else:
            bins = np.delete(bins, merge_idx)

    return bins


def bin_categorical(x: pd.Series, y: pd.Series) -> dict:
    """
    类别型特征分箱：直接按类别值分箱，计算每个类别的坏人率，
    返回 {类别值: 坏人率} 字典（用于后续按坏人率排序、合并小类别）。

    类别型特征不做等频分箱，而是：
    1. 统计每个类别的好坏分布
    2. 低频类别（< min_bin_rate）合并为 "OTHER"
    3. 返回类别→坏人率映射，供 calculate_woe_iv 使用

    参数
    ----
    x : 类别型特征列
    y : TARGET列

    返回
    ----
    cat_map : dict，{原始类别值: 合并后类别标签}
    """
    mask    = x.notna()
    x_valid = x[mask].astype(str)
    y_valid = y[mask]
    n_total = len(x_valid)

    # 统计每个类别的样本量和坏人数
    summary = (pd.DataFrame({'cat': x_valid, 'y': y_valid})
               .groupby('cat')['y']
               .agg(['count', 'sum'])
               .rename(columns={'sum': 'bad'}))
    summary['rate'] = summary['count'] / n_total

    # 低频类别合并为 OTHER
    cat_map = {}
    for cat, row in summary.iterrows():
        if row['rate'] < MIN_BIN_RATE:
            cat_map[cat] = 'OTHER'
        else:
            cat_map[cat] = cat

    return cat_map


# ════════════════════════════════════════════════════════════
#  Part 2：WoE / IV 计算
# ════════════════════════════════════════════════════════════

def calculate_woe_iv(x: pd.Series, y: pd.Series,
                     bins=None, cat_map: dict = None,
                     is_categorical: bool = False) -> pd.DataFrame:
    """
    给定分箱方案，计算每个箱的 WoE 和 IV。

    WoE 公式（每箱 i）：
        WoE_i = ln( (好人_i/总好人) / (坏人_i/总坏人) )
    IV 公式（每箱 i）：
        IV_i  = (好人_i/总好人 - 坏人_i/总坏人) × WoE_i
    总 IV = sum(IV_i)

    WoE 业务含义：
    - WoE > 0：该箱好人占比 > 坏人占比，正向区间（低风险）
    - WoE < 0：该箱坏人占比 > 好人占比，负向区间（高风险）
    - WoE 绝对值越大：该箱对好坏的区分度越强

    注意：缺失值单独作为一个箱处理（NaN箱）

    参数
    ----
    x             : 特征列
    y             : TARGET列
    bins          : 数值型分箱边界（bin_numeric 的返回值）
    cat_map       : 类别型合并映射（bin_categorical 的返回值）
    is_categorical: 是否为类别型特征

    返回
    ----
    woe_df : DataFrame，含列：
             [bin_label, count, bad, good, bad_rate,
              bad_pct, good_pct, woe, iv, feature]
    """
    total_bad  = y.sum()
    total_good = len(y) - total_bad

    # 极端情况：全好人或全坏人，直接返回空
    if total_bad == 0 or total_good == 0:
        return pd.DataFrame()

    rows = []

    if is_categorical:
        # ── 类别型：按 cat_map 分组 ──────────────────────────
        x_mapped = x.astype(str).map(cat_map).fillna('OTHER')
        # 缺失值单独处理
        null_mask = x.isna()
        groups    = sorted(x_mapped[~null_mask].unique())

        for grp in groups:
            mask  = (x_mapped == grp) & (~null_mask)
            n     = mask.sum()
            n_bad = y[mask].sum()
            rows.append({'bin_label': str(grp), 'count': n, 'bad': n_bad})

        # NaN 箱
        if null_mask.sum() > 0:
            rows.append({
                'bin_label': 'NaN',
                'count'    : null_mask.sum(),
                'bad'      : y[null_mask].sum(),
            })
    else:
        # ── 数值型：按 bins 切割 ──────────────────────────────
        null_mask = x.isna()
        x_valid   = x[~null_mask]
        y_valid   = y[~null_mask]
        labels    = pd.cut(x_valid, bins=bins, include_lowest=True)

        for interval, group in y_valid.groupby(labels, observed=True):
            rows.append({
                'bin_label': str(interval),
                'count'    : len(group),
                'bad'      : group.sum(),
            })

        # NaN 箱
        if null_mask.sum() > 0:
            rows.append({
                'bin_label': 'NaN',
                'count'    : null_mask.sum(),
                'bad'      : y[null_mask].sum(),
            })

    # ── 计算 WoE / IV ────────────────────────────────────────
    woe_df = pd.DataFrame(rows)
    if woe_df.empty:
        return woe_df

    woe_df['good']     = woe_df['count'] - woe_df['bad']
    woe_df['bad_rate'] = woe_df['bad'] / woe_df['count']

    # 加平滑值 0.5 防止 0/0（Laplace 平滑）
    woe_df['bad_pct']  = (woe_df['bad']  + 0.5) / (total_bad  + 0.5)
    woe_df['good_pct'] = (woe_df['good'] + 0.5) / (total_good + 0.5)

    woe_df['woe'] = np.log(woe_df['good_pct'] / woe_df['bad_pct'])
    woe_df['iv']  = (woe_df['good_pct'] - woe_df['bad_pct']) * woe_df['woe']

    return woe_df


def _check_monotone(woe_series: pd.Series) -> bool:
    """
    检查 WoE 序列是否单调（递增或递减）。
    评分卡监管要求：数值型特征的 WoE 必须随特征值单调变化，
    以保证评分卡的业务可解释性（分越高风险越低，或反之）。
    """
    vals = woe_series.dropna().values
    if len(vals) <= 2:
        return True
    diffs    = np.diff(vals)
    is_inc   = np.all(diffs >= 0)
    is_dec   = np.all(diffs <= 0)
    return is_inc or is_dec


def enforce_monotone(x: pd.Series, y: pd.Series,
                     bins: np.ndarray) -> np.ndarray:
    """
    通过逐步合并相邻箱来强制 WoE 单调。

    策略：
    1. 计算当前 bins 的 WoE 序列（忽略 NaN 箱）
    2. 若不单调，找 WoE 方向违反的相邻箱对，合并（删去中间边界）
    3. 重复直到单调或只剩2个箱

    这是评分卡开发的核心合规要求：
    监管要求特征与违约概率的关系必须是单调的，
    否则评分卡在某些分段的业务逻辑会自相矛盾。
    """
    current_bins = bins.copy()

    for _ in range(len(bins)):
        if len(current_bins) <= 2:
            break

        woe_df = calculate_woe_iv(x, y, bins=current_bins)
        # 只取非NaN箱做单调性判断
        non_nan = woe_df[woe_df['bin_label'] != 'NaN']
        if non_nan.empty or _check_monotone(non_nan['woe']):
            break

        # 找第一个违反单调性的位置并合并
        woe_vals = non_nan['woe'].values
        diffs    = np.diff(woe_vals)
        # 判断主方向（众数方向）
        main_dir = 1 if (diffs > 0).sum() >= (diffs < 0).sum() else -1
        # 找第一个与主方向相反的 diff
        violations = np.where(diffs * main_dir < 0)[0]
        if len(violations) == 0:
            break
        merge_pos = violations[0] + 1  # 要删除的 bins 内部边界索引
        # current_bins 内部边界从 index 1 到 -1
        # merge_pos 对应 current_bins[merge_pos]（非inf端）
        if 0 < merge_pos < len(current_bins) - 1:
            current_bins = np.delete(current_bins, merge_pos)

    return current_bins


# ════════════════════════════════════════════════════════════
#  Part 3：fit_woe —— 对所有特征拟合分箱 + WoE映射表
# ════════════════════════════════════════════════════════════

def fit_woe(df: pd.DataFrame, feature_cols: list,
            target_col: str = TARGET) -> tuple:
    """
    对筛选后的特征逐一拟合 WoE 编码方案，输出：
      1. woe_map   : dict，{特征名: woe_df}，存储每个特征的分箱→WoE映射
      2. iv_summary: DataFrame，每个特征的总IV值及是否保留

    处理流程（每个特征）：
    ┌─────────────────────────────────────────┐
    │ 1. 判断特征类型（数值/类别）              │
    │ 2. 数值型：等频分箱 → 单调性约束          │
    │    类别型：低频合并 → 计算WoE            │
    │ 3. 计算每箱 WoE / IV                    │
    │ 4. 累加总 IV，判断是否 >= IV_THRESHOLD   │
    └─────────────────────────────────────────┘

    参数
    ----
    df           : 包含所有特征和 TARGET 的 DataFrame
    feature_cols : LGB筛选后的特征列表
    target_col   : 目标变量列名

    返回
    ----
    woe_map    : {feat: woe_df}
    iv_summary : DataFrame [feature, iv, keep]
    """
    y       = df[target_col]
    woe_map = {}
    iv_rows = []

    print(f"\n[WoE] 开始拟合，共 {len(feature_cols)} 个特征")

    for feat in feature_cols:
        x = df[feat]

        # ── 判断特征类型 ──────────────────────────────────
        is_cat = (x.dtype == object or
                  x.dtype.name == 'category' or
                  x.nunique() <= 10)   # 低基数数值也当类别处理

        try:
            if is_cat:
                cat_map = bin_categorical(x, y)
                woe_df  = calculate_woe_iv(
                    x, y, cat_map=cat_map, is_categorical=True
                )
            else:
                # 等频分箱
                raw_bins = bin_numeric(x, y)
                # 强制单调
                mono_bins = enforce_monotone(x, y, raw_bins)
                woe_df    = calculate_woe_iv(x, y, bins=mono_bins)

            if woe_df.empty:
                raise ValueError("WoE计算结果为空")

            # 总 IV（排除 NaN 箱，NaN 箱单独处理不计入IV）
            total_iv = woe_df.loc[
                woe_df['bin_label'] != 'NaN', 'iv'
            ].sum()

            # 存储映射表（附加元信息）
            woe_df['feature']    = feat
            woe_df['is_cat']     = is_cat
            woe_map[feat]        = woe_df

            iv_rows.append({
                'feature' : feat,
                'iv'      : round(total_iv, 6),
                'n_bins'  : len(woe_df[woe_df['bin_label'] != 'NaN']),
                'keep'    : total_iv >= IV_THRESHOLD,
            })

            iv_label = _iv_label(total_iv)
            print(f"  {feat:<45s} IV={total_iv:.4f}  [{iv_label}]"
                  f"  bins={len(woe_df)}")

        except Exception as e:
            print(f"  [WARN] {feat} 拟合失败: {e}")
            iv_rows.append({
                'feature': feat, 'iv': 0.0, 'n_bins': 0, 'keep': False
            })

    iv_summary = (pd.DataFrame(iv_rows)
                  .sort_values('iv', ascending=False)
                  .reset_index(drop=True))

    kept   = iv_summary['keep'].sum()
    dropped = len(iv_summary) - kept
    print(f"\n[WoE] 拟合完成: 保留 {kept} 个 / 剔除 {dropped} 个 (IV < {IV_THRESHOLD})")

    return woe_map, iv_summary


def _iv_label(iv: float) -> str:
    """IV 强度标签，方便日志阅读"""
    if iv < 0.02:  return '无预测力'
    if iv < 0.1:   return '弱'
    if iv < 0.3:   return '中'
    return '强'


# ════════════════════════════════════════════════════════════
#  Part 4：transform_woe + 持久化 + 主入口
# ════════════════════════════════════════════════════════════

def transform_woe(df: pd.DataFrame, woe_map: dict,
                  iv_summary: pd.DataFrame) -> pd.DataFrame:
    """
    将原始特征替换为 WoE 值，供 LR 建模使用。

    原理：
    - LR 要求特征与 log-odds 呈线性关系
    - WoE 编码后，每个分箱用该箱的 WoE 值表示
    - WoE 本身就是 log(好人占比/坏人占比)，与 log-odds 天然线性
    - 缺失值箱（NaN箱）有独立的 WoE 值，不会产生新的缺失

    参数
    ----
    df          : 含原始特征的 DataFrame
    woe_map     : fit_woe 返回的映射字典
    iv_summary  : fit_woe 返回的IV汇总，用于筛选保留特征

    返回
    ----
    df_woe : 只含 WoE 编码后特征（+ TARGET + ID）的 DataFrame
    """
    from src.config import TARGET, ID_COL

    # 只转换 IV 筛选后保留的特征
    keep_feats = iv_summary.loc[iv_summary['keep'], 'feature'].tolist()
    print(f"\n[WoE Transform] 转换 {len(keep_feats)} 个保留特征")

    df_woe = df[[ID_COL, TARGET]].copy() if TARGET in df.columns \
             else df[[ID_COL]].copy()

    for feat in keep_feats:
        if feat not in woe_map:
            continue
        woe_df = woe_map[feat]
        is_cat = woe_df['is_cat'].iloc[0]
        x      = df[feat]

        if is_cat:
            # 类别型：用 bin_label（合并后类别）→ WoE 的字典映射
            lbl_woe = woe_df.set_index('bin_label')['woe'].to_dict()
            nan_woe   = lbl_woe.get('NaN', 0.0)
            # 被合并为 OTHER 的低频类别，用 OTHER 箱的 WoE 兜底
            # 若没有 OTHER 箱（全部类别都高频），退化为 0
            other_woe = lbl_woe.get('OTHER', 0.0)
            x_str  = x.astype(str)
            mapped = x_str.map(lbl_woe)
            # 原始缺失 → NaN 箱的 WoE
            mapped = mapped.copy()
            mapped.loc[x.isna()] = nan_woe
            # 未映射到的类别（低频 OTHER 类 / 生产新类别）→ OTHER 箱 WoE
            mapped = mapped.fillna(other_woe)
            df_woe[feat + '_WOE'] = mapped.values

        else:
            # 数值型：重建 bins，逐行查找所在区间的 WoE
            non_nan_df = woe_df[woe_df['bin_label'] != 'NaN'].copy()
            nan_woe    = woe_df.loc[
                woe_df['bin_label'] == 'NaN', 'woe'
            ].values
            nan_woe = float(nan_woe[0]) if len(nan_woe) > 0 else 0.0

            # 解析区间字符串 → 重建 bins 数组
            bins = _parse_bins_from_labels(non_nan_df['bin_label'].tolist())
            lbl_woe = dict(zip(non_nan_df['bin_label'], non_nan_df['woe']))

            null_mask = x.isna()
            labels    = pd.cut(x[~null_mask], bins=bins,
                               include_lowest=True).astype(str)
            woe_vals  = np.full(len(x), np.nan)
            woe_vals[~null_mask] = labels.map(lbl_woe).fillna(0.0).values
            woe_vals[null_mask]  = nan_woe
            df_woe[feat + '_WOE'] = woe_vals

    print(f"  WoE 特征矩阵 shape: {df_woe.shape}")
    return df_woe


def _parse_bins_from_labels(labels: list) -> np.ndarray:
    """
    从 pd.cut 生成的区间字符串（如 '(-inf, 0.5]'）反解出 bins 数组。
    用于 transform 阶段重建切割边界。
    策略：移除括号后按逗号分割，逐个解析数值端点。
    """
    import re
    edges = set()
    for lbl in labels:
        # 去掉 ( ) [ ] 后按逗号分割两端
        inner = re.sub(r'[\(\)\[\]]', '', lbl).strip()
        for part in inner.split(','):
            part = part.strip()
            if part == '-inf':
                edges.add(-np.inf)
            elif part in ('inf', '+inf'):
                edges.add(np.inf)
            else:
                try:
                    edges.add(float(part))
                except ValueError:
                    pass
    return np.array(sorted(edges))


# ── 持久化：保存 / 加载 WoE 映射表 ──────────────────────────

def save_woe(woe_map: dict, iv_summary: pd.DataFrame) -> None:
    """
    将 WoE 映射表和 IV 汇总保存到 outputs/features/，
    供后续 transform（生产推理）或复盘使用。
    """
    # 合并所有特征的 woe_df 保存为一个 CSV
    all_woe = pd.concat(woe_map.values(), ignore_index=True)
    woe_path = os.path.join(FEATURE_DIR, 'woe_map.csv')
    iv_path  = os.path.join(FEATURE_DIR, 'iv_summary.csv')

    all_woe.to_csv(woe_path, index=False)
    iv_summary.to_csv(iv_path, index=False)
    print(f"  WoE 映射表已保存: {woe_path}")
    print(f"  IV 汇总已保存:   {iv_path}")


def load_woe() -> tuple:
    """
    从 CSV 重新加载 woe_map 和 iv_summary（推理/复盘用）。
    """
    woe_path = os.path.join(FEATURE_DIR, 'woe_map.csv')
    iv_path  = os.path.join(FEATURE_DIR, 'iv_summary.csv')

    all_woe    = pd.read_csv(woe_path)
    iv_summary = pd.read_csv(iv_path)

    woe_map = {
        feat: grp.reset_index(drop=True)
        for feat, grp in all_woe.groupby('feature')
    }
    print(f"  WoE 映射表已加载: {len(woe_map)} 个特征")
    return woe_map, iv_summary


# ════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════

def run_woe_encoding(df: pd.DataFrame,
                     selected_features: list) -> tuple:
    """
    WoE 编码完整流程入口，供 main.py Step5 调用。

    流程：
    1. fit_woe：对 selected_features 拟合分箱 + WoE/IV
    2. 保存 woe_map.csv / iv_summary.csv
    3. transform_woe：将原始特征转为 WoE 编码矩阵
    4. 返回 (df_woe, woe_map, iv_summary)

    参数
    ----
    df                : 含所有特征和 TARGET 的 DataFrame
    selected_features : LGB 筛选后的 Top-N 特征列表

    返回
    ----
    df_woe     : WoE 编码后的特征矩阵（含 TARGET）
    woe_map    : 分箱→WoE 映射字典（供 scorecard.py 使用）
    iv_summary : IV 汇总 DataFrame（含 keep 标记）
    """
    print("\n" + "=" * 55)
    print("[Step 5] WoE 编码")
    print("=" * 55)

    # Step 5-1: 拟合
    woe_map, iv_summary = fit_woe(df, selected_features)

    # Step 5-2: 保存
    save_woe(woe_map, iv_summary)

    # Step 5-3: 转换
    df_woe = transform_woe(df, woe_map, iv_summary)

    print("\n[Step 5] WoE 编码完成")
    print(f"  最终特征数: {df_woe.shape[1] - 2}")   # 减去 ID + TARGET
    print("=" * 55)

    return df_woe, woe_map, iv_summary
