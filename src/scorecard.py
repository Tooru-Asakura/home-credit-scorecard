"""
评分卡转换模块
流程：LR系数 → 每箱分值 → 标准评分卡表格 → 单人评分函数
"""
import numpy as np
import pandas as pd
import os

from src.config import PDO, BASE_SCORE, BASE_ODDS, SCORECARD_DIR, TARGET, ID_COL


# ════════════════════════════════════════════════════════════
#  Part 1：评分卡转换公式
# ════════════════════════════════════════════════════════════

def calc_scorecard_params(pdo: float = PDO,
                          base_score: float = BASE_SCORE,
                          base_odds: float = BASE_ODDS) -> tuple:
    """
    计算评分卡的两个核心转换参数 factor 和 offset。

    标准评分卡转换公式推导：
    ─────────────────────────────────────────────────────────
    设：
      score = offset - factor × ln(odds)
      odds  = P(好人) / P(坏人) = exp(β₀ + β₁x₁ + ... + βₙxₙ)

    由 PDO 定义（好坏比翻倍时分数增加 PDO 分）：
      score + PDO = offset - factor × ln(2 × odds)
    两式相减：
      PDO = factor × ln(2)
      → factor = PDO / ln(2)

    由基准分定义（好坏比为 base_odds 时分数为 base_score）：
      base_score = offset - factor × ln(base_odds)
      → offset = base_score + factor × ln(base_odds)

    参数
    ----
    pdo        : Points to Double Odds，好坏比翻倍对应的分值增量（默认20）
    base_score : 基准分（默认600）
    base_odds  : 基准好坏比，即好人数/坏人数（默认50）

    返回
    ----
    factor : float，放大系数
    offset : float，基础偏移量
    """
    factor = pdo / np.log(2)
    offset = base_score + factor * np.log(base_odds)

    print(f"[评分卡参数]")
    print(f"  PDO={pdo}, 基准分={base_score}, 基准好坏比={base_odds}")
    print(f"  factor = {factor:.4f}")
    print(f"  offset = {offset:.4f}")

    return factor, offset


def build_scorecard_table(woe_map: dict,
                          coef_df: pd.DataFrame,
                          factor: float,
                          offset: float) -> pd.DataFrame:
    """
    将 LR 系数 + WoE 映射表转换为标准评分卡格式。

    每个分箱的得分公式：
    ─────────────────────────────────────────────────────────
    LR 模型：
      log-odds = β₀ + β₁×WoE₁ + β₂×WoE₂ + ... + βₙ×WoEₙ

    将截距均摊到每个特征：
      β₀ 分摊部分 = β₀ / n_features（每个特征分担相同的基础分）

    每个特征 j 的第 i 个分箱得分：
      score_ji = -(β_j × WoE_ji + β₀/n) × factor + offset/n

    其中负号来自：
      score 越高 → odds 越大 → 越像好人 → 违约概率越低
      而 LR 的 log-odds 越大 → 违约概率越低（与评分方向一致）

    参数
    ----
    woe_map  : fit_woe 返回的 {特征名: woe_df} 字典
    coef_df  : build_coef_table 返回的系数表（含 coef_mean）
    factor   : calc_scorecard_params 返回的放大系数
    offset   : calc_scorecard_params 返回的偏移量

    返回
    ----
    scorecard : DataFrame，含列：
                [feature, bin_label, woe, coef, score,
                 count, bad_rate]
    """
    # 从 coef_df 构建 {WoE列名: 系数} 和 {WoE列名: 原始特征名} 的映射
    coef_lookup = {}
    for _, row in coef_df.iterrows():
        coef_lookup[row['feature']] = row['coef_mean']

    # 获取进入模型的原始特征名（去掉 _WOE 后缀），排除截距行
    model_feats = [
        c.replace('_WOE', '') for c in coef_df['feature'].tolist()
        if c != '_INTERCEPT'
    ]
    n_feats = len(model_feats)

    if n_feats == 0:
        raise ValueError("[评分卡] coef_df 为空，无法生成评分卡")

    # 截距均摊：从 coef_df 提取截距行（由 build_coef_table 写入）
    intercept_rows = coef_df[coef_df['feature'] == '_INTERCEPT']
    intercept      = float(intercept_rows['coef_mean'].values[0]) \
                     if len(intercept_rows) > 0 else 0.0
    # 每个特征分担的截距部分
    intercept_per_feat = intercept / n_feats

    rows = []
    for feat in model_feats:
        woe_col = feat + '_WOE'
        if feat not in woe_map:
            continue

        coef   = coef_lookup.get(woe_col, 0.0)
        woe_df = woe_map[feat].copy()

        for _, bin_row in woe_df.iterrows():
            woe_val = bin_row['woe']

            # 每箱得分 = -(系数 × WoE + 截距均摊) × factor + offset/n
            bin_score = (
                -(coef * woe_val + intercept_per_feat) * factor
                + offset / n_feats
            )

            rows.append({
                'feature'   : feat,
                'bin_label' : bin_row['bin_label'],
                'woe'       : round(woe_val, 4),
                'coef'      : round(coef, 6),
                'score'     : round(bin_score, 2),
                'count'     : int(bin_row['count']),
                'bad_rate'  : round(bin_row['bad_rate'], 4),
                'is_nan_bin': bin_row['bin_label'] == 'NaN',
            })

    scorecard = pd.DataFrame(rows)
    print(f"\n[评分卡] 共生成 {len(model_feats)} 个特征、"
          f"{len(scorecard)} 条分箱得分记录")

    return scorecard


# ════════════════════════════════════════════════════════════
#  Part 2：单人评分函数 + 评分卡校验
# ════════════════════════════════════════════════════════════

def score_single(applicant: dict,
                 scorecard: pd.DataFrame,
                 woe_map: dict) -> dict:
    """
    给定单个申请人的原始特征字典，返回其总评分及各特征得分明细。

    评分流程：
    1. 对每个特征，找到申请人落入的分箱
    2. 查表得到该分箱的分值
    3. 累加所有特征分值 = 总分

    业务约定：
    - 总分越高 → 好人（违约概率越低）
    - 一般设定切分点：如总分 < 500 拒绝，500~600 人工复核，> 600 通过

    参数
    ----
    applicant : dict，{原始特征名: 特征值}（不含 WoE 后缀）
    scorecard : build_scorecard_table 生成的评分卡表
    woe_map   : fit_woe 返回的映射字典

    返回
    ----
    result : dict，含 total_score 和各特征明细列表
    """
    features = scorecard['feature'].unique()
    details  = []
    total    = 0.0

    for feat in features:
        feat_card = scorecard[scorecard['feature'] == feat]
        raw_val   = applicant.get(feat, np.nan)

        # 找到申请人所在分箱
        bin_label = _find_bin(raw_val, feat, woe_map)

        # 查评分卡
        match = feat_card[feat_card['bin_label'] == bin_label]
        if len(match) == 0:
            # 未匹配到分箱（新类别或边界外），用 NaN 箱
            match = feat_card[feat_card['bin_label'] == 'NaN']

        score = float(match['score'].values[0]) if len(match) > 0 else 0.0
        total += score

        details.append({
            'feature'  : feat,
            'raw_value': raw_val,
            'bin_label': bin_label,
            'score'    : round(score, 2),
        })

    return {
        'total_score': round(total, 2),
        'details'    : details,
    }


def _find_bin(value, feat: str, woe_map: dict) -> str:
    """
    根据原始特征值找到其在评分卡中的分箱标签。

    - 缺失值 → 'NaN' 箱
    - 类别型 → 直接匹配 bin_label（已含 OTHER 兜底）
    - 数值型 → 解析区间边界，用 pd.cut 找区间
    """
    import re

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'NaN'

    if feat not in woe_map:
        return 'NaN'

    woe_df   = woe_map[feat]
    is_cat   = bool(woe_df['is_cat'].iloc[0])
    non_nan  = woe_df[woe_df['bin_label'] != 'NaN']

    if is_cat:
        str_val = str(value)
        if str_val in non_nan['bin_label'].values:
            return str_val
        return 'OTHER'
    else:
        # 数值型：从 bin_label 字符串解析边界（与 _parse_bins_from_labels 同逻辑）
        edges = set()
        for lbl in non_nan['bin_label']:
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
        bins = np.array(sorted(edges))

        try:
            label = pd.cut([float(value)], bins=bins,
                           include_lowest=True)[0]
            return str(label) if not pd.isna(label) else 'NaN'
        except Exception:
            return 'NaN'


def validate_scorecard(scorecard: pd.DataFrame,
                       df_raw: pd.DataFrame,
                       woe_map: dict) -> pd.DataFrame:
    """
    用训练集批量计算评分，输出分数分布统计，校验评分卡合理性。

    校验项目：
    1. 分数分布（均值/标准差/分位数）
    2. 好坏人分数分布对比（好人均分 > 坏人均分 才是正确方向）
    3. 各分数段的违约率（KS 曲线的直观呈现）

    参数
    ----
    scorecard : 评分卡表
    df_raw    : 含原始特征和 TARGET 的 DataFrame
    woe_map   : WoE 映射字典

    返回
    ----
    score_df : DataFrame，含每个样本的 total_score 和 TARGET
    """
    MAX_VALIDATE_ROWS = 5000   # 最多用5000行做校验，节省时间

    features = scorecard['feature'].unique().tolist()
    sample   = df_raw[features + [TARGET]]
    if len(sample) > MAX_VALIDATE_ROWS:
        sample = sample.sample(MAX_VALIDATE_ROWS, random_state=42)
        print(f"  （已采样 {MAX_VALIDATE_ROWS} 行进行校验）")

    scores = []
    for _, row in sample.iterrows():
        applicant = row[features].to_dict()
        result    = score_single(applicant, scorecard, woe_map)
        scores.append(result['total_score'])

    score_df = pd.DataFrame({
        'score' : scores,
        TARGET  : sample[TARGET].values,
    })

    # ── 分布统计 ──────────────────────────────────────────
    good_scores = score_df.loc[score_df[TARGET] == 0, 'score']
    bad_scores  = score_df.loc[score_df[TARGET] == 1, 'score']

    print(f"\n  全量分布：均值={score_df['score'].mean():.1f}  "
          f"std={score_df['score'].std():.1f}  "
          f"min={score_df['score'].min():.1f}  "
          f"max={score_df['score'].max():.1f}")
    print(f"  好人均分：{good_scores.mean():.1f}  "
          f"坏人均分：{bad_scores.mean():.1f}  "
          f"差值：{good_scores.mean() - bad_scores.mean():.1f}")

    direction_ok = good_scores.mean() > bad_scores.mean()
    print(f"  方向校验：{'✓ 好人分 > 坏人分（正确）' if direction_ok else '✗ 方向异常！好人分 < 坏人分'}")

    # ── 分段违约率（10等分）────────────────────────────────
    score_df['score_bin'] = pd.qcut(
        score_df['score'], q=10, duplicates='drop'
    )
    seg = (score_df.groupby('score_bin', observed=True)[TARGET]
           .agg(['count', 'mean'])
           .rename(columns={'mean': 'bad_rate'})
           .reset_index())
    seg['bad_rate'] = seg['bad_rate'].round(4)
    print(f"\n  分数段违约率（应随分数升高而单调递减）：")
    print(seg.to_string(index=False))

    return score_df


# ════════════════════════════════════════════════════════════
#  Part 3：持久化 + 主入口
# ════════════════════════════════════════════════════════════

def save_scorecard(scorecard: pd.DataFrame,
                   score_df: pd.DataFrame = None) -> None:
    """
    保存评分卡表格到 outputs/scorecard/。

    输出文件：
    - scorecard.csv : 标准评分卡（特征/分箱/WoE/系数/得分）
    - score_dist.csv: 训练集分数分布（含 TARGET，用于复盘）
    """
    sc_path = os.path.join(SCORECARD_DIR, 'scorecard.csv')
    scorecard.to_csv(sc_path, index=False)
    print(f"  评分卡已保存: {sc_path}")

    if score_df is not None:
        dist_path = os.path.join(SCORECARD_DIR, 'score_dist.csv')
        score_df.to_csv(dist_path, index=False)
        print(f"  分数分布已保存: {dist_path}")


def print_scorecard(scorecard: pd.DataFrame,
                    top_features: int = None) -> None:
    """
    格式化打印评分卡，便于终端查阅。

    参数
    ----
    scorecard    : 评分卡表
    top_features : 只打印前 N 个特征（None = 全部）
    """
    features = scorecard['feature'].unique()
    if top_features:
        features = features[:top_features]

    print("\n" + "═" * 70)
    print(f"{'特征':<35} {'分箱区间':<25} {'WoE':>7} {'得分':>7} {'坏率':>7}")
    print("═" * 70)

    for feat in features:
        feat_rows = scorecard[scorecard['feature'] == feat]
        print(f"\n▶ {feat}")
        for _, row in feat_rows.iterrows():
            nan_flag = ' [NaN箱]' if row['is_nan_bin'] else ''
            print(f"  {'':3s}{str(row['bin_label']):<32}"
                  f"{row['woe']:>7.4f}"
                  f"{row['score']:>8.1f}"
                  f"{row['bad_rate']:>8.4f}"
                  f"{nan_flag}")

    print("═" * 70)


def run_scorecard(df_raw: pd.DataFrame,
                  woe_map: dict,
                  coef_df: pd.DataFrame,
                  iv_summary: pd.DataFrame,
                  validate: bool = True) -> pd.DataFrame:
    """
    评分卡生成完整流程入口，供 main.py Step7 调用。

    流程：
    1. 计算 factor / offset（PDO=20, 基准分=600, 好坏比=50）
    2. build_scorecard_table：LR系数 × WoE → 每箱得分
    3. 保存评分卡 CSV
    4. （可选）validate_scorecard：批量评分 + 分布校验
    5. 打印评分卡样本

    参数
    ----
    df_raw     : 含原始特征和 TARGET 的 DataFrame（用于校验）
    woe_map    : run_woe_encoding 返回的 WoE 映射字典
    coef_df    : run_lr_modeling 返回的系数表
    iv_summary : run_woe_encoding 返回的 IV 汇总
    validate   : 是否执行批量评分校验（数据量大时可设 False 跳过）

    返回
    ----
    scorecard : 标准评分卡 DataFrame
    """
    print("\n" + "=" * 55)
    print("[Step 7] 评分卡生成")
    print("=" * 55)

    # Step 7-1: 计算转换参数
    factor, offset = calc_scorecard_params()

    # Step 7-2: 生成评分卡
    scorecard = build_scorecard_table(woe_map, coef_df, factor, offset)

    # Step 7-3: 保存
    if validate:
        # 批量评分（用训练集的原始特征，注意这里只做分布校验，不是测试集）
        score_df = validate_scorecard(scorecard, df_raw, woe_map)
        save_scorecard(scorecard, score_df)
    else:
        save_scorecard(scorecard)
        score_df = None

    # Step 7-4: 打印前5个特征的评分卡
    print("\n[评分卡预览] 前5个特征：")
    print_scorecard(scorecard, top_features=5)

    print("\n[Step 7] 评分卡生成完成")
    print(f"  输出目录: {SCORECARD_DIR}")
    print("=" * 55)

    return scorecard

