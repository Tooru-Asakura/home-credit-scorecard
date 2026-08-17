import numpy as np
import pandas as pd
from tqdm import tqdm


# ════════════════════════════════════════════════════════
#  主表特征
# ════════════════════════════════════════════════════════
def feat_application(df: pd.DataFrame) -> pd.DataFrame:
    """
    主表（application_train）衍生特征。
    原始字段已在 preprocessing.clean_application() 中完成异常值处理：
      - DAYS_EMPLOYED=365243（未就业占位符）已替换为 NaN
      - AGE_YEARS / DAYS_EMPLOYED_YEARS 已在清洗时生成
    """
    df = df.copy()

    # ── 收入负债类 ────────────────────────────────────────────
    # DTI（债务收入比）：贷款金额 / 年收入
    # 值越大 → 还款压力越大 → 违约风险越高
    # 对应 FICO 评分中"欠债金额"维度（权重30%）
    df['CREDIT_INCOME_RATIO']  = df['AMT_CREDIT']  / (df['AMT_INCOME_TOTAL'] + 1)

    # 月还款收入比：年化还款额 / 年收入（类似 payment-to-income ratio）
    # 值越大 → 月现金流压力越大 → 容错空间越窄
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)

    # 贷款期限估算：贷款总额 / 年化还款额（单位：年）
    # 值越大 → 期限越长 → 长期信用暴露风险越高
    df['CREDIT_TERM']          = df['AMT_CREDIT']  / (df['AMT_ANNUITY']      + 1)

    # 套现信号：贷款金额 / 货物价格
    # 正常消费贷 ≈ 1；若 >> 1 说明贷款额远超商品价值，疑似套现行为
    df['CREDIT_GOODS_RATIO']   = df['AMT_CREDIT']  / (df['AMT_GOODS_PRICE']  + 1)

    # 家庭人均收入：年收入 / 家庭成员数
    # 反映家庭真实可支配能力，比总收入更能体现还款余力
    df['INCOME_PER_PERSON']    = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)

    # ── 就业稳定性 ────────────────────────────────────────────
    # 就业年限占年龄比：工作年限 / 年龄
    # 比值高 → 职业生涯大部分时间都在稳定就业 → 信用稳定性强
    # 注意：DAYS_EMPLOYED 中 365243 已替换 NaN，无效值不参与计算
    df['EMPLOYED_TO_AGE_RATIO'] = (
        df['DAYS_EMPLOYED_YEARS'] / (df['AGE_YEARS'] + 1)
    )

    # ── 申请材料完整度 ────────────────────────────────────────
    # 提交文件数量：FLAG_DOCUMENT_2 ~ FLAG_DOCUMENT_21 的求和
    # 材料越少 → 信息不透明 → 欺诈/违约风险可能越高
    doc_cols     = [c for c in df.columns if 'FLAG_DOCUMENT' in c]
    contact_cols = ['FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE',
                    'FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL']
    df['DOCUMENT_COUNT'] = df[doc_cols].sum(axis=1)

    # 联系方式数量：提供的联系渠道越多 → 可触达性强 → 催收容易 → 风险偏低
    df['CONTACT_COUNT']  = df[[c for c in contact_cols
                                if c in df.columns]].sum(axis=1)

    # ── 社交圈风险传染 ────────────────────────────────────────
    # 30天社交圈违约率：申请人身边熟人中近30天逾期的比例
    # 行为金融学：社交圈的信用风险具有一定传染性，可作为软信息
    df['SOCIAL_CIRCLE_DEF_RATE_30'] = (
        df['DEF_30_CNT_SOCIAL_CIRCLE'] / (df['OBS_30_CNT_SOCIAL_CIRCLE'] + 1)
    )
    # 60天社交圈违约率：同上，更长观察窗口，捕捉更严重/持续的逾期
    df['SOCIAL_CIRCLE_DEF_RATE_60'] = (
        df['DEF_60_CNT_SOCIAL_CIRCLE'] / (df['OBS_60_CNT_SOCIAL_CIRCLE'] + 1)
    )

    # ── EXT_SOURCE 第三方评分交叉 ────────────────────────────
    # EXT_SOURCE_1/2/3：来自外部征信机构的综合评分（类似 FICO 子分项）
    # 是数据集中预测力最强的字段，需多角度挖掘其信息量
    ext = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

    # 三源均值：综合外部信用评分，缺失时自动忽略
    df['EXT_SOURCE_MEAN'] = df[ext].mean(axis=1)

    # 三源标准差：评分离散度越大 → 不同机构评价分歧越大 → 信用画像不稳定
    df['EXT_SOURCE_STD']  = df[ext].std(axis=1)

    # 三源乘积：同时惩罚所有维度偏低的情况（任一极低则乘积极小）
    df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']

    # 两两交叉：捕捉两个评分源之间的协同效应
    # EXT_SOURCE_1×2：两机构评分同时高 → 信用一致性强
    df['EXT_SOURCE_1_2']  = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    # EXT_SOURCE_1×3：同上
    df['EXT_SOURCE_1_3']  = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    # EXT_SOURCE_2×3：EXT_SOURCE_2 是最强单特征，与3的交叉尤为重要
    df['EXT_SOURCE_2_3']  = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']

    # EXT_SOURCE_2 × 年龄：外部评分与年龄的交互
    # 年长且评分高 → 长期信用积累可信；年轻但评分高 → 快速建立信用
    # 两者乘积可放大"老+好"组合的信号
    df['EXT_SOURCE_2_AGE'] = df['EXT_SOURCE_2'] * df['AGE_YEARS']

    return df


# ════════════════════════════════════════════════════════
#  bureau + bureau_balance
# ════════════════════════════════════════════════════════
def feat_bureau(bureau: pd.DataFrame,
                bureau_bal: pd.DataFrame) -> pd.DataFrame:
    """
    外部征信聚合特征（两级聚合）。
    数据层次：bureau_balance（月度快照）→ bureau（单笔借据）→ application（申请人）
    这是 Home Credit 数据集中最能反映"历史信用行为"的外部数据源。
    """

    # ── Level 1: bureau_balance → SK_ID_BUREAU ──────────────
    # bureau_balance 记录每笔外部贷款每个月的还款状态（月度快照）
    # STATUS 含义：C=结清, X=未知, 0=无逾期, 1~5=逾期1~5个月以上
    bureau_bal = bureau_bal.copy()
    bureau_bal['STATUS_NUM'] = bureau_bal['STATUS'].map(
        {'C':0,'X':0,'0':0,'1':1,'2':2,'3':3,'4':4,'5':5}
    )
    # IS_DPD：该月是否存在逾期（DPD = Days Past Due）
    # STATUS_NUM >= 1 即表示存在逾期
    bureau_bal['IS_DPD'] = (bureau_bal['STATUS_NUM'] >= 1).astype(int)

    bb_agg = bureau_bal.groupby('SK_ID_BUREAU').agg(
        # 该笔贷款的月度记录总条数（贷款存续时长）
        BB_MONTH_COUNT = ('MONTHS_BALANCE', 'count'),
        # 累计逾期月数：反映历史逾期的频次
        BB_DPD_COUNT   = ('IS_DPD',         'sum'),
        # 逾期月度比率：逾期月数/总月数 → 逾期习惯性程度
        BB_DPD_RATE    = ('IS_DPD',         'mean'),
        # 最严重逾期等级（1~5月以上）：反映历史最差信用记录
        BB_STATUS_MAX  = ('STATUS_NUM',     'max'),
        # 平均逾期等级：反映整体还款质量
        BB_STATUS_MEAN = ('STATUS_NUM',     'mean'),
    ).reset_index()

    # ── Level 2: bureau → SK_ID_CURR ────────────────────────
    # bureau 记录申请人在各外部金融机构的每一笔贷款信息
    bureau = bureau.copy()
    bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')

    # 单笔外部贷款的使用率：已用额度 / 批准额度
    # 越高 → 该笔贷款资金压力越大（类似信用卡利用率）
    bureau['BUREAU_CREDIT_UTIL'] = (
        bureau['AMT_CREDIT_SUM_DEBT'] / (bureau['AMT_CREDIT_SUM'] + 1)
    )
    # 该笔贷款是否仍处于激活状态（Active = 尚未结清）
    bureau['IS_ACTIVE']      = (bureau['CREDIT_ACTIVE'] == 'Active').astype(int)
    # DAYS_CREDIT 是负数（相对申请日往前的天数），取绝对值方便理解
    bureau['DAYS_CREDIT_ABS']= -bureau['DAYS_CREDIT']

    agg = bureau.groupby('SK_ID_CURR').agg(
        # 历史外部借贷笔数：笔数多 → 信用活跃，也可能表示过度负债
        BUREAU_COUNT             = ('SK_ID_BUREAU',          'count'),
        # 当前仍有效的贷款笔数：正在承担的外部债务数量
        BUREAU_ACTIVE_COUNT      = ('IS_ACTIVE',             'sum'),
        # 活跃贷款比例：当前负债面/历史总借贷面
        BUREAU_ACTIVE_RATE       = ('IS_ACTIVE',             'mean'),
        # 平均/最大单笔外部贷款额度：反映信用额度水平
        BUREAU_CREDIT_SUM_MEAN   = ('AMT_CREDIT_SUM',        'mean'),
        BUREAU_CREDIT_SUM_MAX    = ('AMT_CREDIT_SUM',        'max'),
        # 外部贷款未偿债务总额：用于计算 BUREAU_DEBT_INCOME_RATIO（跨表特征）
        BUREAU_DEBT_SUM          = ('AMT_CREDIT_SUM_DEBT',   'sum'),
        # 平均单笔未偿债务：反映每笔贷款平均欠款水平
        BUREAU_DEBT_MEAN         = ('AMT_CREDIT_SUM_DEBT',   'mean'),
        # 平均逾期金额：有逾期的贷款平均拖欠多少钱
        BUREAU_OVERDUE_MEAN      = ('AMT_CREDIT_SUM_OVERDUE','mean'),
        # 历史最大单笔逾期金额：极端风险事件的捕捉
        BUREAU_MAX_OVERDUE       = ('AMT_CREDIT_MAX_OVERDUE','max'),
        # 平均/最大借贷记录年龄（天）：越久远 → 信用积累越长
        BUREAU_DAYS_CREDIT_MEAN  = ('DAYS_CREDIT_ABS',       'mean'),
        BUREAU_DAYS_CREDIT_MAX   = ('DAYS_CREDIT_ABS',       'max'),
        # 平均/最大信用使用率（已用/批准）：越高 → 外部负债越紧张
        BUREAU_UTIL_MEAN         = ('BUREAU_CREDIT_UTIL',    'mean'),
        BUREAU_UTIL_MAX          = ('BUREAU_CREDIT_UTIL',    'max'),
        # 历史展期次数总和：展期 = 无力按时还款请求延期，是软性违约信号
        BUREAU_PROLONG_SUM       = ('CNT_CREDIT_PROLONG',    'sum'),
        # 月度逾期率均值（来自 bureau_balance 的两级聚合）：
        # 对申请人名下所有外部贷款的月度逾期率取均值，反映长期还款习惯
        BUREAU_BB_DPD_RATE_MEAN  = ('BB_DPD_RATE',           'mean'),
        # 月度逾期总次数（来自 bureau_balance）：绝对逾期频率
        BUREAU_BB_DPD_COUNT_SUM  = ('BB_DPD_COUNT',          'sum'),
        # 历史最严重逾期等级（来自 bureau_balance）：捕捉最坏的信用记录
        BUREAU_BB_STATUS_MAX     = ('BB_STATUS_MAX',         'max'),
    ).reset_index()

    return agg


# ════════════════════════════════════════════════════════
#  installments_payments
# ════════════════════════════════════════════════════════
def feat_installments(inst: pd.DataFrame) -> pd.DataFrame:
    """
    分期还款行为特征（installments_payments）。
    每行记录一次应还/实还事件，是最直接反映"还款纪律"的行为数据。
    DAYS 字段均为负数（相对申请日往前的天数），数值越大（绝对值越小）越近期。
    """
    inst = inst.copy()

    # DPD（Days Past Due）：实际还款日 - 应还款日，正数 = 逾期天数
    # 用 clip(0) 将提前还款（负数）截断为0，只保留逾期部分
    inst['DPD']  = (inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']).clip(lower=0)

    # DBD（Days Before Due）：应还款日 - 实际还款日，正数 = 提前天数
    # 提前还款是良好还款习惯的体现，DBD 大 → 偿债意愿强
    inst['DBD']  = (inst['DAYS_INSTALMENT'] - inst['DAYS_ENTRY_PAYMENT']).clip(lower=0)

    # 还款比例：实还金额 / 应还金额
    # = 1 → 全额还款；< 1 → 欠缴；> 1 → 超额还款（提前偿本）
    inst['PAYMENT_RATIO'] = inst['AMT_PAYMENT'] / (inst['AMT_INSTALMENT'] + 1)

    # 还款缺口：应还 - 实还（正数表示少还的绝对金额）
    # 用于捕捉欠款金额的量级，比 PAYMENT_RATIO 更直观
    inst['PAYMENT_DIFF']  = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']

    # 是否逾期（二值）：DPD > 0 即为逾期，用于统计逾期频率
    inst['IS_LATE']       = (inst['DPD'] > 0).astype(int)

    # 是否少还（二值）：少还 ≠ 逾期（可能按时少还），反映资金缺口习惯
    inst['IS_UNDERPAY']   = (inst['PAYMENT_DIFF'] > 0).astype(int)

    # ── 全量聚合（全部历史记录）────────────────────────────
    agg_all = inst.groupby('SK_ID_CURR').agg(
        # 历史还款记录总条数（贷款活跃程度）
        INST_COUNT              = ('SK_ID_PREV',      'count'),
        # 平均逾期天数：整体还款延迟程度
        INST_DPD_MEAN           = ('DPD',             'mean'),
        # 最大单次逾期天数：捕捉最严重的一次违约行为
        INST_DPD_MAX            = ('DPD',             'max'),
        # 逾期天数总和：综合衡量历史逾期严重程度
        INST_DPD_SUM            = ('DPD',             'sum'),
        # 平均提前还款天数：越大 → 还款习惯越好
        INST_DBD_MEAN           = ('DBD',             'mean'),
        # 逾期率：逾期次数 / 总还款次数（还款纪律核心指标）
        INST_LATE_RATE          = ('IS_LATE',         'mean'),
        # 逾期次数绝对值
        INST_LATE_COUNT         = ('IS_LATE',         'sum'),
        # 少还率：少还次数 / 总还款次数
        INST_UNDERPAY_RATE      = ('IS_UNDERPAY',     'mean'),
        # 平均还款比例：整体还款完整性
        INST_PAYMENT_RATIO_MEAN = ('PAYMENT_RATIO',   'mean'),
        # 最低还款比例：捕捉还款最差的那一次（极端值）
        INST_PAYMENT_RATIO_MIN  = ('PAYMENT_RATIO',   'min'),
        # 平均单次欠款金额：欠款习惯的量级
        INST_PAYMENT_DIFF_MEAN  = ('PAYMENT_DIFF',    'mean'),
        # 历史累计欠款总额：总负债缺口
        INST_PAYMENT_DIFF_SUM   = ('PAYMENT_DIFF',    'sum'),
    ).reset_index()

    # ── 近6个月窗口聚合（时序趋势特征）──────────────────────
    # DAYS_INSTALMENT >= -180 → 最近180天内的还款记录
    # 近期行为比远期行为更能预测未来违约（行为漂移捕捉）
    inst_6m = inst[inst['DAYS_INSTALMENT'] >= -180]
    agg_6m  = inst_6m.groupby('SK_ID_CURR').agg(
        # 近6个月平均逾期天数
        INST_RECENT_DPD_MEAN  = ('DPD',           'mean'),
        # 近6个月逾期率
        INST_RECENT_LATE_RATE = ('IS_LATE',        'mean'),
        # 近6个月平均还款比例
        INST_RECENT_PAY_RATIO = ('PAYMENT_RATIO',  'mean'),
    ).reset_index()
    agg_6m.columns = ['SK_ID_CURR'] + [
        c + '_6M' for c in agg_6m.columns[1:]
    ]

    agg = agg_all.merge(agg_6m, on='SK_ID_CURR', how='left')

    # ── 趋势特征（近期 vs 全量）──────────────────────────────
    # 近6M平均DPD / 全量平均DPD：
    # > 1 → 近期逾期在恶化（比历史均值更差），是最强的风险预警信号
    # < 1 → 近期还款改善（可能经过债务整合或收入回升）
    agg['INST_DPD_TREND']  = (
        agg['INST_RECENT_DPD_MEAN_6M'] / (agg['INST_DPD_MEAN'] + 1e-5)
    )
    # 近6M逾期率 / 全量逾期率：同上，从频率维度判断趋势方向
    agg['INST_LATE_TREND'] = (
        agg['INST_RECENT_LATE_RATE_6M'] / (agg['INST_LATE_RATE'] + 1e-5)
    )

    return agg


# ════════════════════════════════════════════════════════
#  credit_card_balance
# ════════════════════════════════════════════════════════
def feat_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    """
    信用卡使用行为特征（credit_card_balance）。
    每行是一张信用卡某个月的账单快照，反映"循环信贷"的使用习惯。
    信用卡行为在 FICO 评分体系中占30%权重（"欠款金额"维度），
    是最能体现持续负债压力与消费习惯的行为特征之一。
    """
    cc = cc.copy()

    # 信用卡使用率（Utilization Rate）：当月余额 / 信用额度
    # FICO 评分最关键指标之一：> 30% 开始扣分，> 90% 极高风险
    # 高利用率 → 资金紧张 → 更容易违约
    cc['CC_UTILIZATION']   = (
        cc['AMT_BALANCE'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
    )

    # 当月还款比例：本期实还 / 本期余额
    # = 1 → 全额还款（最优）；低比例 → 滚动负债，利息累积
    # 长期低比例还款 → 负债滚雪球风险
    cc['CC_PAYMENT_RATIO'] = (
        cc['AMT_PAYMENT_CURRENT'] / (cc['AMT_BALANCE'] + 1)
    )

    # 最低还款比率：本期实还 / 应还总额（含最低还款要求）
    # 仅支付最低还款额 → 偿债意愿弱 / 流动性不足的信号
    # 监管层面：长期仅还最低额是高风险客群特征
    cc['CC_MIN_PAY_RATIO'] = (
        cc['AMT_PAYMENT_CURRENT'] / (cc['AMT_PAYMENT_TOTAL_CURRENT'] + 1)
    )

    # 该账单月是否存在逾期（SK_DPD = statement中的逾期天数）
    cc['CC_IS_DPD']        = (cc['SK_DPD'] > 0).astype(int)

    agg = cc.groupby('SK_ID_CURR').agg(
        # 信用卡账单总条数（信用卡活跃使用时长）
        CC_COUNT              = ('SK_ID_PREV',                'count'),
        # 平均/最大月末余额：持续高余额 → 长期高负债状态
        CC_BALANCE_MEAN       = ('AMT_BALANCE',               'mean'),
        CC_BALANCE_MAX        = ('AMT_BALANCE',               'max'),
        # 平均信用额度：额度越高通常代表信用资质越好
        CC_LIMIT_MEAN         = ('AMT_CREDIT_LIMIT_ACTUAL',   'mean'),
        # 平均/最大信用卡使用率：核心风险指标
        # 平均值反映日常负债习惯，最大值捕捉历史峰值压力
        CC_UTILIZATION_MEAN   = ('CC_UTILIZATION',            'mean'),
        CC_UTILIZATION_MAX    = ('CC_UTILIZATION',            'max'),
        # 平均还款比例：整体偿还完整性
        CC_PAYMENT_RATIO_MEAN = ('CC_PAYMENT_RATIO',          'mean'),
        # 平均最低还款比率：长期仅还最低额是强风险信号
        CC_MIN_PAY_RATIO_MEAN = ('CC_MIN_PAY_RATIO',          'mean'),
        # 平均/最大逾期天数：信用卡逾期记录
        CC_DPD_MEAN           = ('SK_DPD',                    'mean'),
        CC_DPD_MAX            = ('SK_DPD',                    'max'),
        # 账单逾期率：逾期账单月数 / 总账单月数
        CC_IS_DPD_RATE        = ('CC_IS_DPD',                 'mean'),
        # 平均总消费金额：消费活跃度，过高可能超出还款能力
        CC_DRAWINGS_MEAN      = ('AMT_DRAWINGS_CURRENT',      'mean'),
    ).reset_index()

    return agg


# ════════════════════════════════════════════════════════
#  POS_CASH_balance
# ════════════════════════════════════════════════════════
def feat_pos(pos: pd.DataFrame) -> pd.DataFrame:
    """
    POS机消费贷 & 现金贷月度还款状态特征（POS_CASH_balance）。
    每行是一笔 Home Credit 历史贷款某个月的还款状态快照，
    反映客户在 HC 自身产品线上的还款纪律，是内部行为数据（与 bureau 外部数据互补）。
    SK_DPD：当月逾期天数；SK_DPD_DEF：当月"严重逾期"天数（监管定义的违约级别）
    """
    pos = pos.copy()

    # 普通逾期标志：SK_DPD > 0 即存在任意逾期
    # 用于统计逾期频率（宽口径）
    pos['POS_IS_DPD']     = (pos['SK_DPD']     > 0).astype(int)

    # 严重逾期标志：SK_DPD_DEF > 0，通常为监管定义的"违约"级别逾期
    # 比普通 DPD 更严格，是更强的违约信号（窄口径）
    pos['POS_IS_DPD_DEF'] = (pos['SK_DPD_DEF'] > 0).astype(int)

    # 贷款完成进度：1 - 剩余期数 / 总期数
    # 接近0 → 刚开始还款；接近1 → 即将还清
    # 进度越高而仍逾期 → 说明还款能力持续不足，风险更大
    pos['POS_PROGRESS']   = (
        1 - pos['CNT_INSTALMENT_FUTURE'] / (pos['CNT_INSTALMENT'] + 1)
    )

    agg = pos.groupby('SK_ID_CURR').agg(
        # 历史 POS/现金贷账单总条数（产品使用频率）
        POS_COUNT           = ('SK_ID_PREV',      'count'),
        # 平均/最大逾期天数：整体还款延迟水平 & 历史最差单月
        POS_DPD_MEAN        = ('SK_DPD',          'mean'),
        POS_DPD_MAX         = ('SK_DPD',          'max'),
        # 普通逾期月份比率：逾期频率（宽口径）
        POS_IS_DPD_RATE     = ('POS_IS_DPD',      'mean'),
        # 严重逾期月份比率：监管级违约频率（窄口径，权重更高）
        POS_IS_DPD_DEF_RATE = ('POS_IS_DPD_DEF',  'mean'),
        # 平均还款进度：反映申请人当前在 HC 贷款中处于的还款阶段
        POS_PROGRESS_MEAN   = ('POS_PROGRESS',    'mean'),
        # 月度快照总条数（贷款存续月数，反映与 HC 的关系深度）
        POS_MONTHS_COUNT    = ('MONTHS_BALANCE',  'count'),
    ).reset_index()

    return agg


# ════════════════════════════════════════════════════════
#  previous_application
# ════════════════════════════════════════════════════════
def feat_previous(prev: pd.DataFrame) -> pd.DataFrame:
    """
    HC 历史申请记录特征（previous_application）。
    记录该申请人过去在 Home Credit 提交的每一笔贷款申请（无论结果如何），
    反映"申请行为模式"——短时间内频繁申请是过度负债或欺诈的信号，
    与外部征信的 hard inquiry 概念类似，在 FICO 中占10%权重（新申请维度）。
    """
    prev = prev.copy()

    # 历史实批金额 / 申请金额：
    # > 1 → 批额超出申请（少见）；< 1 → 被砍额（风控收紧）
    # 长期被砍额 → HC 内部风控历史上认为该客户资质不足
    prev['PREV_CREDIT_APP_RATIO'] = (
        prev['AMT_CREDIT'] / (prev['AMT_APPLICATION'] + 1)
    )

    # 申请结果二值化（用于后续聚合计算比率）
    # Approved：历次申请被批准（HC 认可该客户风险）
    prev['IS_APPROVED'] = (prev['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
    # Refused：历次申请被拒绝（HC 评估后认为风险过高）
    prev['IS_REFUSED']  = (prev['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    # Canceled：客户主动取消（可能是比价后放弃，也可能是条件不满意）
    prev['IS_CANCELED'] = (prev['NAME_CONTRACT_STATUS'] == 'Canceled').astype(int)

    # 首付比例：首付金额 / 批准贷款额
    # 首付比例越高 → 客户自有资金越充足 → 风险越低
    # 零首付消费贷则此值为0
    prev['DOWN_PAY_RATIO'] = (
        prev['AMT_DOWN_PAYMENT'] / (prev['AMT_CREDIT'] + 1)
    )

    agg = prev.groupby('SK_ID_CURR').agg(
        # 历史申请总笔数：笔数多 → 信贷需求旺盛，也可能是多头借贷信号
        PREV_COUNT             = ('SK_ID_PREV',             'count'),
        # 历史被批笔数 / 被拒笔数 / 取消笔数（绝对值）
        PREV_APPROVED_COUNT    = ('IS_APPROVED',            'sum'),
        PREV_REFUSED_COUNT     = ('IS_REFUSED',             'sum'),
        PREV_CANCELED_COUNT    = ('IS_CANCELED',            'sum'),
        # 历史通过率：被批次数 / 总申请次数（HC 内部信用认可度）
        PREV_APPROVED_RATE     = ('IS_APPROVED',            'mean'),
        # 历史拒绝率：被拒次数 / 总申请次数
        # 拒绝率高 → HC 长期认为该客户风险偏高，是强负向特征
        # 与 CREDIT_INCOME_RATIO 交叉后构成跨表特征 REFUSED_RATE_CREDIT_CROSS
        PREV_REFUSED_RATE      = ('IS_REFUSED',             'mean'),
        # 历史平均/最大批款金额：反映 HC 对该客户的历史授信水平
        PREV_CREDIT_MEAN       = ('AMT_CREDIT',             'mean'),
        PREV_CREDIT_MAX        = ('AMT_CREDIT',             'max'),
        # 历史平均申请金额：客户自身的借款需求量级
        PREV_APP_MEAN          = ('AMT_APPLICATION',        'mean'),
        # 平均批款/申请比：历史上 HC 批额相对申请额的折扣程度
        PREV_CREDIT_APP_RATIO  = ('PREV_CREDIT_APP_RATIO',  'mean'),
        # 历史平均年化还款额：过去贷款的还款压力参考
        PREV_ANNUITY_MEAN      = ('AMT_ANNUITY',            'mean'),
        # 平均首付比例：历史自有资金参与程度
        PREV_DOWN_PAY_RATIO    = ('DOWN_PAY_RATIO',         'mean'),
        # 平均合同期数（月）：历史贷款的平均期限，越长说明偏好长期借贷
        PREV_CNT_PAYMENT_MEAN  = ('CNT_PAYMENT',            'mean'),
        # 平均决策日期距申请日天数（负数，绝对值越小 = 越近期有申请）
        PREV_DAYS_DECISION_MEAN= ('DAYS_DECISION',          'mean'),
    ).reset_index()

    return agg


# ════════════════════════════════════════════════════════
#  跨表交叉特征（merge后执行）
# ════════════════════════════════════════════════════════
def feat_cross(df: pd.DataFrame) -> pd.DataFrame:
    """
    跨表交叉特征：在全表 merge 完成后执行，利用多张表的聚合结果进行组合。
    核心思想：单一特征描述一个维度的风险，两个相关维度的乘积能放大共同信号，
    捕捉"多重风险叠加"的客群，这类客群的违约率通常显著高于单维度高风险客群。
    """
    df = df.copy()

    # ── 外部总负债 / 年收入（杠杆率）────────────────────────
    # 将 bureau 的债务总量（BUREAU_DEBT_SUM）归一化到收入维度
    # 类比宏观杠杆率概念：外部负债 / 年收入越高 → 偿债能力越弱
    # 比 CREDIT_INCOME_RATIO 更全面——后者仅含本次申请，此特征涵盖所有外部债务
    if 'BUREAU_DEBT_SUM' in df.columns:
        df['BUREAU_DEBT_INCOME_RATIO'] = (
            df['BUREAU_DEBT_SUM'] / (df['AMT_INCOME_TOTAL'] + 1)
        )

    # ── 历史被拒率 × 当前贷款压力（双重负向信号）────────────
    # PREV_REFUSED_RATE：HC 历史上认为该客户风险高（机构视角）
    # CREDIT_INCOME_RATIO：当前这笔贷款对其收入的压力（客观负担）
    # 两者都高 → 既有历史不良记录，当前又承压过重，违约概率极高
    if 'PREV_REFUSED_RATE' in df.columns:
        df['REFUSED_RATE_CREDIT_CROSS'] = (
            df['PREV_REFUSED_RATE'] * df['CREDIT_INCOME_RATIO']
        )

    # ── 信用卡高利用率 × 外部征信逾期（双重资金紧张信号）────
    # CC_UTILIZATION_MEAN：信用卡额度已被高度占用（短期流动性紧张）
    # BUREAU_BB_DPD_RATE_MEAN：外部贷款历史上频繁逾期（长期还款习惯差）
    # 两者叠加 → 既缺乏流动性缓冲，又有不良还款习惯，是强复合风险信号
    if 'CC_UTILIZATION_MEAN' in df.columns and 'BUREAU_BB_DPD_RATE_MEAN' in df.columns:
        df['CC_UTIL_BUREAU_DPD_CROSS'] = (
            df['CC_UTILIZATION_MEAN'] * df['BUREAU_BB_DPD_RATE_MEAN']
        )

    # ── 工作稳定性 × 外部信用评分（正向叠加信号）────────────
    # DAYS_EMPLOYED_YEARS：就业年限越长 → 收入来源越稳定
    # EXT_SOURCE_2：外部综合信用评分越高 → 历史信用越好
    # 两者乘积越大 → 同时具备"稳定收入"和"良好信用"的优质客群
    # 反之，短期就业且评分低 → 乘积极小，高风险客群
    if 'EXT_SOURCE_2' in df.columns:
        df['EMPLOYED_EXT2_CROSS'] = (
            df['DAYS_EMPLOYED_YEARS'] * df['EXT_SOURCE_2']
        )

    return df


# ════════════════════════════════════════════════════════
#  汇总：合并所有表
# ════════════════════════════════════════════════════════
def build_features(data: dict) -> pd.DataFrame:
    """
    主函数：读取data字典，输出融��后的特征表
    data = load_all() 的返回值
    """
    print("=" * 50)
    print("[特征工程] 开始构建特征")

    # Step1: 主表清洗 + 衍生
    from src.preprocessing import clean_application, encode_categoricals
    app = clean_application(data['application'])
    app = feat_application(app)
    print(f"  主表特征: {app.shape}")

    # Step2: 各子表聚合
    feats = {
        'bureau' : feat_bureau(data['bureau'], data['bureau_balance']),
        'inst'   : feat_installments(data['installments']),
        'cc'     : feat_credit_card(data['credit_card']),
        'pos'    : feat_pos(data['pos']),
        'prev'   : feat_previous(data['previous']),
    }
    for name, f in feats.items():
        print(f"  {name:10s} 特征: {f.shape}")

    # Step3: 全部 left join 到主表
    df = app.copy()
    for name, f in feats.items():
        df = df.merge(f, on='SK_ID_CURR', how='left')

    # Step4: 跨表交叉特征
    df = feat_cross(df)

    # Step5: 类别编码 + 缺失值填充
    from src.preprocessing import encode_categoricals, fill_missing
    df = encode_categoricals(df)
    df = fill_missing(df)

    print(f"[特征工程] 完成，最终 shape: {df.shape}")
    print("=" * 50)
    return df