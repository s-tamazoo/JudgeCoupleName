import pandas as pd
import lightgbm as lgb
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def ReadTrainingData():

    # 2つのCSVファイルを読み込んで結合
    df1 = pd.read_csv(os.path.abspath("data/カップリング名リスト.csv"), encoding="utf-8")
    df2 = pd.read_csv(os.path.abspath("data/4文字名詞リスト.csv"), encoding="utf-8")
    df = pd.concat([df1, df2], axis=0)

    # データフレームの成形
    df = df.fillna(0)
    df['1文字目'] = df['1文字目'].astype(int)
    df['2文字目'] = df['2文字目'].astype(int)
    df['3文字目'] = df['3文字目'].astype(int)
    df['4文字目'] = df['4文字目'].astype(int)

    return df


def TrainModel(df):

    # 特徴行列と目標変数を分割
    X = df[['1文字目', '2文字目', '3文字目', '4文字目']]
    y = df['カプ名フラグ']

    # データをトレーニングセットとテストセットに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # LightGBMのモデルを初期化
    model = lgb.LGBMClassifier()

    # モデルをトレーニング
    model.fit(X_train, y_train)

    # テストデータで予測
    y_pred = model.predict(X_test)

    # モデルの評価
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # 学習結果を出力
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", classification_rep)

    return model


def main():

    df = ReadTrainingData()
    model = TrainModel(df)

    # モデルを保存
    joblib.dump(model, 'lightgbm_model.pkl')


if __name__ == "__main__":
    main()