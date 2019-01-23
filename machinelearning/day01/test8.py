from scipy.stats import pearsonr
import pandas as pd

def correlation_demo():
    """
    pearsonr相关系数演示
    :return:
    """
    data = pd.read_csv("factor_returns.csv")

    data = data.iloc[:, 1:-2]

    correlation = pearsonr(data["pe_ratio"], data["pb_ratio"])
    print(correlation)

    return None


if __name__ == '__main__':
    correlation_demo()