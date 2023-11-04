
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取fraud.csv文件
data = pd.read_csv('fraud.csv', encoding='gbk')


# fraud_elems字典，包含要保留的列名
fraud_elems = {
    '0': ['诈骗公私财物数额较大', '诈骗公私财物数额巨大', '诈骗公私财物数额特别巨大'],
    '1': ['诈骗手段恶劣|危害严重', '通过短信|电话|互联网|广播电视|报刊杂志发布虚假信息诈骗', '诈骗救灾|抢险|防汛|优抚|扶贫|移民|救济|医疗款物', '诈骗残疾人|老年人|丧失劳动能力人的财物',
          '造成被害人自杀|精神失常|其他严重后果的', '冒充国家机关工作人员实施诈骗', '组织|指挥电网网络诈骗犯罪团伙', '境外实施电信网络诈骗',
          '曾因电信网络诈骗受过刑事处罚', '利用电话追呼系统等技术严重干扰公安工作','属于诈骗集团首要分子'],
    '2': ['诈骗近亲属的财物', '案发前自动将赃物归还被害人', '没有参与分赃|获赃较少且不是主犯', '确因生活所迫|学习|治病急需诈骗', '多次实施诈骗'],
    '3': ['诈骗金额不足1万元', '诈骗金额1-3万元', '诈骗金额3-10万元', '诈骗金额10-20万元', '诈骗金额20-50万元', '诈骗金额50-150万元', '诈骗金额150-300万元', '诈骗金额超过300万元'],
    '4': ['犯罪既遂', '犯罪未遂', '犯罪中止', '犯罪预备'],
    '5': ['主犯', '犯罪集团首要分子', '一般累犯', '惯犯', '犯罪前有劣迹', '前科', '认罪态度不好', '手段恶劣', '放任危害结果', '犯罪后逃跑'],
    '6': ['从犯', '立功', '重大立功', '残疾人犯', '坦白', '一贯表现好', '自首', '当庭自愿认罪', '被害人有过错', '主动取得被害人谅解',
          '与被害人和解', '家庭有困难', '认罪态度好', '初犯', '偶犯', '罪行较轻且自首', '退赃退赔', '赃款赃物全部被追缴', '没有造成损害的中止犯']
}

# 获取fraud_elems中的所有列名
valid_columns = set(col for cols in fraud_elems.values() for col in cols)

# 遍历数据的列名，如果不在valid_columns中则删除该列
for column in data.columns:
    if column != '刑期' and column not in valid_columns:
        data.drop(column, axis=1, inplace=True)

# 删除那些列中的所有元素都为0的列
data = data.loc[:, (data != 0).any(axis=0)]
#筛选有矛盾的行
valid_columns = ['诈骗公私财物数额较大', '诈骗公私财物数额巨大', '诈骗公私财物数额特别巨大']
# 判断指定列至少有两个为1的行
mask = data[valid_columns].sum(axis=1) >= 2
# 删除符合条件的行
data = data[~mask]
# 遍历数据每行，检查指定列至少有两个为1的行
valid_columns = ['诈骗金额不足1万元', '诈骗金额1-3万元', '诈骗金额3-10万元', '诈骗金额10-20万元',
                 '诈骗金额20-50万元', '诈骗金额50-150万元', '诈骗金额150-300万元', '诈骗金额超过300万元']
# 判断指定列至少有两个为1的行
mask = data[valid_columns].sum(axis=1) >= 2
# 删除符合条件的行
data = data[~mask]
# 遍历数据每行，检查指定列至少有两个为1的行
valid_columns = ['犯罪既遂', '犯罪未遂', '犯罪中止']
# 判断指定列至少有两个为1的行
mask = data[valid_columns].sum(axis=1) >= 2
# 删除符合条件的行
data = data[~mask]
data = data[~((data['主犯'] == 1) & (data['从犯'] == 1))]
data = data[~((data['初犯'] == 1) & (data['偶犯'] == 1))]
# 输出处理后的DataFrame
data = data.reset_index(drop=True)
data.to_csv('new_fraud.csv', index=False, encoding='gbk')


# 打乱数据划分训练验证测试集
data = data.sample(frac=1, random_state=42)  # 使用42作为随机种子，确保可重复性
train_ratio = 0.6
val_ratio = 0.15
test_ratio = 0.25
train, test = train_test_split(data, test_size=val_ratio + test_ratio, random_state=42)
val, test = train_test_split(test, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)
train.to_csv('train_set.csv', index=False)
val.to_csv('val_set.csv', index=False)
test.to_csv('test_set.csv', index=False)
