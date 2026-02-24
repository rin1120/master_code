import requests
import pandas as pd
import time
from tqdm import tqdm

# --- 設定 ---
API_KEY = "8afe621325a73567c3f21793619389a2"  # ← あなたのTMDB APIキーをここに
BASE_URL = "https://api.themoviedb.org/3"
TARGET_MOVIE_COUNT = 6000
CSV_FILE_NAME = "movies_2000.csv"

# --- ターゲットジャンル＆制作会社 ---
TARGET_GENRES = [
    "Adventure", "Fantasy", "Drama", "Action", "Comedy",
    "Thriller", "Crime", "Romance", "Family", "Science Fiction"
]

TARGET_COMPANIES = [
    "Columbia Pictures", "Warner Bros",
    "Universal Pictures", "Twentieth Century Fox Film Corporation"
]

# --- 映画IDを収集 ---
def get_movie_ids(limit):
    ids = set()
    page = 1

    print("🔎 映画ID取得中...")
    with tqdm(total=limit, desc="ID収集") as pbar:
        while len(ids) < limit:
            res = requests.get(f"{BASE_URL}/discover/movie", params={
                "api_key": API_KEY,
                "sort_by": "release_date.desc",
                "language": "en-US",
                "page": page
            }).json()
            new_ids = [movie["id"] for movie in res.get("results", [])]
            for mid in new_ids:
                if mid not in ids:
                    ids.add(mid)
                    pbar.update(1)
                    if len(ids) >= limit:
                        break
            page += 1
            time.sleep(0.2)
    return list(ids)[:limit]

# --- 映画詳細を取得＆ワンホット化 ---
def get_movie_row(movie_id):
    res = requests.get(f"{BASE_URL}/movie/{movie_id}", params={
        "api_key": API_KEY,
        "language": "en-US"
    }).json()

    # 初期化
    row = {"id": movie_id}
    for genre in TARGET_GENRES:
        row[f"【{genre}】"] = 0
    for company in TARGET_COMPANIES:
        row[f"【{company}】"] = 0

    # ジャンル処理
    for g in res.get("genres", []):
        if g["name"] in TARGET_GENRES:
            row[f"【{g['name']}】"] = 1

    # 制作会社処理
    for c in res.get("production_companies", []):
        if c["name"] in TARGET_COMPANIES:
            row[f"【{c['name']}】"] = 1

    return row

# --- メイン処理 ---
def main():
    movie_ids = get_movie_ids(TARGET_MOVIE_COUNT)
    print("🎬 映画データ収集中...")
    all_rows = []

    for mid in tqdm(movie_ids):
        try:
            row = get_movie_row(mid)
            all_rows.append(row)
            time.sleep(0.2)
        except Exception as e:
            print(f"⚠️ エラー（{mid}）: {e}")

    df = pd.DataFrame(all_rows)
    df.to_csv(CSV_FILE_NAME, index=False)
    print(f"✅ 完了！{CSV_FILE_NAME} に保存しました。")

if __name__ == "__main__":
    main()
