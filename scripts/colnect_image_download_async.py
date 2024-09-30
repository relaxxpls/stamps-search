import asyncio
import aiohttp
import aiofiles
import pandas as pd
import os
from aiohttp import ClientSession
from asyncio import Semaphore
from tqdm.asyncio import tqdm

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
MAX_CONCURRENCY = 200
MAX_RETRIES = 3


async def download_image(
    session: ClientSession, semaphore: Semaphore, i: int, image_url: str
):
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(
                    image_url, headers=headers, cookies=cookies
                ) as response:
                    response.raise_for_status()
                    content = await response.read()

                    async with aiofiles.open(f"{img_dir}/{i}.jpg", "wb") as f:
                        await f.write(content)

                return
            except aiohttp.ClientError as e:
                if attempt == MAX_RETRIES - 1:
                    print(
                        f"Failed to download image {i}.jpg after {MAX_RETRIES} attempts",
                        image_url,
                    )
                    print(e)
                else:
                    await asyncio.sleep(1)  # Wait before retrying


async def main():
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    downloaded_images = set(
        int(file.split(".")[0]) for file in os.listdir(img_dir) if file.endswith(".jpg")
    )

    df = pd.read_csv("stanley_gibbons_colnect.csv")
    df = df[~df.index.isin(downloaded_images)]
    total_images = len(df)
    print(f"Total images to download: {total_images}")
    print(f"Images already downloaded: {len(downloaded_images)}")

    semaphore = Semaphore(MAX_CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        tasks = [
            download_image(session, semaphore, i, row["image_url"])
            for i, row in df.iterrows()
        ]

        for task in tqdm.as_completed(
            tasks, total=total_images, desc="Downloading images"
        ):
            await task

    print("Download finished.")


if __name__ == "__main__":
    asyncio.run(main())
