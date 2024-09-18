import pandas as pd
import requests
import os
from multiprocessing.pool import ThreadPool


def download_image(i: int, image_url: str):
    try:
        response = requests.get(image_url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        })
        response.raise_for_status()

        with open(f'images/{i}.jpg', 'wb') as f:
            f.write(response.content)

        print(f"Downloaded image {i}.jpg", image_url)

    except requests.exceptions.HTTPError as e:
        print(f"Failed to download image {i}.jpg", image_url)
        print(e)


def main():
    if not os.path.exists('data/images'):
        os.makedirs('data/images')

    downloaded_images = set()
    for file in os.listdir('data/images'):
        if file.endswith('.jpg'):
            downloaded_images.add(int(file.split('.')[0]))

    df = pd.read_csv('colnect.csv')
    df = df[~df.index.isin(downloaded_images)]
    print(f"Total images to download: {len(df)}")
    print(f"Images already downloaded: {len(downloaded_images)}")

    print("Starting the download...")
    ThreadPool(10).starmap(download_image, [(i, row['image_url']) for i, row in df.iterrows()])
    print("Download finished.")


if __name__ == "__main__":
    main()
