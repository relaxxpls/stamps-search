import pandas as pd
import requests
import os
from multiprocessing.pool import ThreadPool

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    )
}
cookies = {
    "cnv2sess": "m41vjpare5aj6vurrhhgk2jn4j",
    "cnv2user2": "lumgtbv1akg48kwcg4kokkwsokcssow",
    "cnv2_forum_u": "379920",
    "cnv2_forum_sid": "bf8702857e6e8c11775968d6acf2a294",
}
img_dir = "data/images"


def download_image(i: int, image_url: str):
    try:
        response = requests.get(image_url, headers=headers, cookies=cookies)
        response.raise_for_status()

        with open(f"{img_dir}/{i}.jpg", "wb") as f:
            f.write(response.content)

        print(f"Downloaded image {i}.jpg", image_url)

    except requests.exceptions.HTTPError as e:
        print(f"Failed to download image {i}.jpg", image_url)
        print(e)


def main():
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    downloaded_images = set()
    for file in os.listdir(img_dir):
        if file.endswith(".jpg"):
            downloaded_images.add(int(file.split(".")[0]))

    df = pd.read_csv("stanley_gibbons_colnect.csv")
    df = df[~df.index.isin(downloaded_images)]
    print(f"Total images to download: {len(df)}")
    print(f"Images already downloaded: {len(downloaded_images)}")

    print("Starting the download...")
    ThreadPool(10).starmap(
        download_image, [(i, row["image_url"]) for i, row in df.iterrows()]
    )
    print("Download finished.")


if __name__ == "__main__":
    main()
