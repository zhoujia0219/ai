from scipy.stats import pearsonr

def correlation_demo():
    """
    相关系数演示
    :return:
    """
    # 1.获取数据
    factor = ['pe_ratio', 'pb_ratio', 'market_cap', 'return_on_asset_net_profit', 'du_return_on_equity', 'ev',
              'earnings_per_share', 'revenue', 'total_expense']

    data = factor.iloc[:, 1:-2]
    correlation = pearsonr(data["pe_ratio"], data["pb_ratio"])
    print(correlation)

    return None

if __name__ == '__main__':
    correlation_demo()