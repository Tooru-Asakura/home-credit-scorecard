import os
import pandas as pd
from src.config import DATA_RAW, FILES


def load_data(name: str) -> pd.DataFrame:
    """
    按名称读取单张表
    name: FILES字典中的key
    """
    path = os.path.join(DATA_RAW, FILES[name])
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    df = pd.read_csv(path)
    print(f"[load] {name:20s} -> shape: {df.shape}")
    return df


def load_all() -> dict:
    """
    读取全部表，返回字典
    返回: {'application': df, 'bureau': df, ...}
    """
    data = {}
    for name in FILES:
        data[name] = load_data(name)
    return data