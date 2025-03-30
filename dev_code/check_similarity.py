# ----------------------------------------------------------------------------
# コンテンツ２つの類似度を計算するプログラム
# ----------------------------------------------------------------------------

import numpy as np
import pandas as pd

# パラメータ
file_path = "500_movies.csv"   # 適宜修正
cid1 = 19                     # 1つ目のコンテンツID (ユーザーが指定)
cid2 = 300                     # 2つ目のコンテンツID (ユーザーが指定)

# CSV読み込み
df = pd.read_csv(file_path)

# コンテンツベクトルの準備
N = len(df.columns) - 1  # 属性数
cont_vector = df.set_index('id').values.tolist()
cont_vector_array = [np.array(vec) for vec in cont_vector]

def compute_content_distance(cid_a, cid_b):
    """
    2つのコンテンツID (cid_a, cid_b) のベクトルを取り出し、
    ユークリッド距離を計算して返す。
    """
    # IDは1からスタートする想定のため、配列indexはcid - 1
    vect_a = cont_vector_array[cid_a - 1]
    vect_b = cont_vector_array[cid_b - 1]
    dist = np.linalg.norm(vect_a - vect_b)
    return dist

def main():
    # 2つのコンテンツIDを指定しているので、直接距離計算
    distance = compute_content_distance(cid1, cid2)
    print(f"Distance between content {cid1} and {cid2} = {distance}")

if __name__ == "__main__":
    main()
