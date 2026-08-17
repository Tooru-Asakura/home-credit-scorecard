import numpy as np
import pandas as pd


def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    """
    主表清洗：
    1. 修复已知异常值
    2. 衍生基础时间特征
    3. 处理极少数异常类别
    """
    df = df.copy()

    # DAYS_EMPLOYED=365243 是未就业占位符，替换为NaN
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # 时间字段转为正数（年）
    df['AGE_YEARS']               = -df['DAYS_BIRTH']      / 365
    df['DAYS_EMPLOYED_YEARS']     = -df['DAYS_EMPLOYED']   / 365
    df['DAYS_REGISTRATION_YEARS'] = -df['DAYS_REGISTRATION'] / 365
    df['DAYS_ID_PUBLISH_YEARS']   = -df['DAYS_ID_PUBLISH'] / 365

    # 性别XNA（仅4条）→ NaN
    df['CODE_GENDER'].replace('XNA', np.nan, inplace=True)

    # 金额字段负值截断为0
    amt_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT',
                'AMT_ANNUITY', 'AMT_GOODS_PRICE']
    for col in amt_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    类别变量编码：LabelEncoding
    说明：LGB可直接处理整数编码的类别变量
          后续LR阶段会用WoE替换，此处只做基础编码
    """
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    cat_cols = df.select_dtypes('object').columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].fillna('missing')
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    缺失值填充策略：
    - 子表聚合的计数/求和类特征：填0（无记录=0次）
    - 其余数值特征：填-999（缺失标记，对树模型友好）
    """
    df = df.copy()

    count_kws = ['COUNT', 'SUM', '_NUM']
    count_cols = [c for c in df.columns
                  if any(kw in c for kw in count_kws)
                  and c not in ['SK_ID_CURR', 'TARGET']]

    df[count_cols] = df[count_cols].fillna(0)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    remain   = [c for c in num_cols
                if c not in count_cols + ['SK_ID_CURR', 'TARGET']]
    df[remain] = df[remain].fillna(-999)

    return df