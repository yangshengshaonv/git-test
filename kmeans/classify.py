import pandas as pd
import numpy as np
def add_label(filepath):
    # 读取数据
    data = pd.read_csv(filepath)

    # 定义区间
    a, b, c, d, e = 0, 26, 56,  104, 180  # 请根据你的实际数据来调整这些值
    if filepath=='../row_data/train_set.csv':
        # 筛选数据并保存到 CSV
        data[(data['刑期'] > a) & (data['刑期'] <= b)].to_csv('../ctgan_syn_data/row_data/Category_0_Data.csv', index=False)
        data[(data['刑期'] > b) & (data['刑期'] <= c)].to_csv('../ctgan_syn_data/row_data/Category_1_Data.csv', index=False)
        data[(data['刑期'] > c) & (data['刑期'] <= d)].to_csv('../ctgan_syn_data/row_data/Category_2_Data.csv', index=False)
        data[(data['刑期'] > d) & (data['刑期'] <= e)].to_csv('../ctgan_syn_data/row_data/Category_3_Data.csv', index=False)
    if filepath == '../row_data/val_set.csv':
        # 筛选数据并保存到 CSV
        data[(data['刑期'] > a) & (data['刑期'] <= b)].to_csv('Category_0_val_Data.csv',index=False)
        data[(data['刑期'] > b) & (data['刑期'] <= c)].to_csv('Category_1_val_Data.csv',index=False)
        data[(data['刑期'] > c) & (data['刑期'] <= d)].to_csv('Category_2_val_Data.csv',index=False)
        data[(data['刑期'] > d) & (data['刑期'] <= e)].to_csv('Category_3_val_Data.csv',index=False)
    if filepath == '../row_data/test_set.csv':
        # 筛选数据并保存到 CSV
        data[(data['刑期'] > a) & (data['刑期'] <= b)].to_csv('Category_0_test_Data.csv',index=False)
        data[(data['刑期'] > b) & (data['刑期'] <= c)].to_csv('Category_1_test_Data.csv',index=False)
        data[(data['刑期'] > c) & (data['刑期'] <= d)].to_csv('Category_2_test_Data.csv',index=False)
        data[(data['刑期'] > d) & (data['刑期'] <= e)].to_csv('Category_3_test_Data.csv',index=False)
        # data[(data['刑期'] > e) & (data['刑期'] <= f)].to_csv('../ctgan_syn_data/row_data/Category_4_Data.csv',
        # index=False) data[(data['刑期'] > f) & (data['刑期'] <= g)].to_csv(
        # '../ctgan_syn_data/row_data/Category_5_Data.csv', index=False) 定义条件和分类标签
    conditions = [
        (data['刑期'] > a) & (data['刑期'] <= b),
        (data['刑期'] > b) & (data['刑期'] <= c),
        (data['刑期'] > c) & (data['刑期'] <= d),
        (data['刑期'] > d) & (data['刑期'] <= e)
        # (data['刑期'] > e) & (data['刑期'] <= f),
        # (data['刑期'] > f) & (data['刑期'] <= g)
    ]
    # labels = ['0', '1', '2', '3', '4', '5']
    labels = ['0', '1', '2', '3']
    # 使用 np.select 为每个数据点赋予分类标签
    data['类别'] = np.select(conditions, labels, default='其他')
    dir=filepath.split('/')[-1].split('.')[0]
    # 保存到 CSV 文件中
    data.to_csv(f'with_category_{dir}_data.csv', index=False)

filepaths=['../row_data/train_set.csv','../row_data/val_set.csv','../row_data/test_set.csv']
for filepath in filepaths:
    add_label(filepath)

# 分类是从小到大排列的
