# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv('train.csv')

# "Sex","Embarked"を数字へ置換
df['Sex'] = df['Sex'].replace({'male':0,'female':1})
df['Embarked'] = df['Embarked'].replace({'S': 0,'C':1,'Q':2})
#　欠損値の置き換え
df.fillna({'Age':df.Age.median(), 'Embarked':0},inplace=True)
# 家族構成
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
# "Cabin"の削除
df = df.drop(["Name", "SibSp", "Parch", "Ticket", "Cabin"], axis=1)

# データをトレーニングデータ，テストデータに分ける
df_train = df[:800]
df_test = df[800:]

# データ予測
from sklearn.ensemble import RandomForestClassifier

df_train_data = df_train.values
xs = df_train_data[:, 2:]
y = df_train_data[:, 1]

forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(xs, y)

df_test_data = df_test.values
xs_test = df_test_data[:, 2:]
output = forest.predict(xs_test)

# 実データの取得
output_true = df_test["Survived"].values

# 予測結果と実データの比較
compare_out = 0
for i in range(len(output)):
    if output[i] == output_true[i]:
        compare_out = compare_out + 1

result = compare_out / len(output) * 100
print(result)
