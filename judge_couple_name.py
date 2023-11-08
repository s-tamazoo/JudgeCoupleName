import re
import joblib
import pykakasi
import pandas as pd

def ConvertVowel(text) -> str:
    """母音に変換する"""

    kakasi = pykakasi.kakasi()

    # 母音以外の残したい文字(っ, ん)を記号にして逃がす
    text = re.sub(r'ん', '★', text)
    text = re.sub(r'っ', '☆', text)

    # 不要な記号を除去
    text = re.sub(r'？', '', text)
    text = re.sub(r'ー', '', text)

    # ひらがなをローマ字に変換
    kakasi.setMode('H', 'a')
    kakasi.setMode('K', 'a')
    kakasi.setMode('J', 'a')
    conversion = kakasi.getConverter()
    text = conversion.do(text)

    # 子音を除去
    text = re.sub(r'b|c|d|f|g|h|j|k|l|m|n|p|q|r|s|t|v|w|x|y|z', '', text)

    # 母音以外の残したい文字(っ, ん)を英字(t, n)に変換
    text = re.sub(r'a', '1', text)
    text = re.sub(r'i', '2', text)
    text = re.sub(r'u', '3', text)
    text = re.sub(r'e', '4', text)
    text = re.sub(r'o', '5', text)
    text = re.sub(r'★', '6', text)
    text = re.sub(r'☆', '7', text)

    # 4文字以下の場合0で埋める
    while len(text) < 4:
        text = text + "0"

    return [int(char) for char in text]


def main():

    # カプ名を入力
    while True:
        cup_name = input("4文字のひらがなを入力してください: ")
        if re.match(r'^[あ-ん]{4}$', cup_name):
            print("入力が有効です。")
            break
        else:
            print("無効な入力です。正確に4文字のひらがなを入力してください。")
    name_boin = ConvertVowel(cup_name)

    # モデルを読み込む
    model = joblib.load('model.pkl')

    # テストデータフレームを作成
    X_test = pd.DataFrame({
        '1文字目': [name_boin[0]],  # 1文字目列のデータ
        '2文字目': [name_boin[1]],  # 2文字目列のデータ
        '3文字目': [name_boin[2]],  # 3文字目列のデータ
        '4文字目': [name_boin[3]]   # 4文字目列のデータ
    })

    # 学習済みモデルを使用して確率で出力
    probabilities = model.predict_proba(X_test)

    # 各サンプルに対するクラス1に属する確率を表示
    for i, prob in enumerate(probabilities):
        print(f"{cup_name}のカップリング名っぽさは {int(prob[1]*100)}% です")


if __name__ == "__main__":
    main()