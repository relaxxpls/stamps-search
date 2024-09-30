import time
import itertools
import json

import pandas as pd

from selenium import webdriver
from bs4 import BeautifulSoup

cookies = {
    "cnv2sess": "m41vjpare5aj6vurrhhgk2jn4j",
    "cnv2user2": "lumgtbv1akg48kwcg4kokkwsokcssow",
    "cnv2_forum_u": "379920",
    "cnv2_forum_sid": "bf8702857e6e8c11775968d6acf2a294",
}

driver = webdriver.Chrome()

for cookie in cookies:
    cookie_item = {
        "domain": "colnect.com",
        "name": cookie,
        "value": cookies[cookie],
    }
    driver.execute_cdp_cmd("Network.setCookie", cookie_item)

# Disable network tracking
driver.execute_cdp_cmd("Network.disable", {})

img_placeholder_url = "i.colnect.net/items/thumb/none-stamps.jpg"
keys_ignore = [
    "",
    "paper",
    "expiry_date",
    "variants",
    "buy_now",
    "related_items",
    "score",
    "gum",
    "overprints",
    "print_run",
    "printing",
]


def get_colnect_url(year: int, page: int):
    colnect_url = f"https://colnect.com/en/stamps/list/year/{year}/catalog/438-Stanley_Gibbons/sort/by_issue_date"
    return colnect_url if page == 1 else f"{colnect_url}/page/{page}"


def scrap_page(url: str):
    time.sleep(2.5)
    driver.get(url)
    time.sleep(0.3)

    current_url = driver.current_url
    if url != current_url:
        raise Exception(f"Finished scrapping the last page: {current_url}")

    soup = BeautifulSoup(driver.page_source, features="lxml")

    slow_down = soup.find(string="Slow Down!")
    if slow_down is not None:
        print(f"Slow down! {url}")
        # retry after 3 seconds
        time.sleep(3)
        return scrap_page(url)

    empty_list = soup.find("h3", string="The list is empty!")
    if empty_list is not None:
        raise Exception(f"The page is empty! {url}")

    items_parent = soup.find("div", id="plist_items")

    if items_parent is None:
        raise Exception(f"Maybe blocked, exiting... {url}")

    items = []
    for child in items_parent.findChildren("div", recursive=False):
        item = {}
        image_url = child.find("img")["data-src"][2:]
        if image_url == img_placeholder_url:
            continue

        item["colnect_url"] = "https://colnect.com" + child.find("a")["href"]
        item["image_url"] = "https://i.colnect.net/b" + image_url[15:]
        item["title"] = child.find("h2").text

        children_table = (
            child.findChildren("div", class_="i_d")[0]
            .findChildren("dl")[0]
            .findChildren(recursive=False)
        )

        for i in range(0, len(children_table), 2):
            key = children_table[i].text
            key = key[:-1]  # Remove semicolon at the end
            key = key.lower().replace(" ", "_").strip()

            if key in keys_ignore:
                continue

            value = children_table[i + 1].text
            item[key] = value

        items.append(item)
        if not image_url.startswith("i.colnect.net"):
            print(f"Image URL: {image_url}")

    return items


def main():
    print("Starting the scrapper...")
    df = pd.DataFrame()

    with open("init_state.json", "r") as file:
        init_state = json.load(file)

    for year in range(init_state["year"], 2025):
        start = 1 if year > init_state["year"] else init_state["page"]
        for ctr in itertools.count(start=start):
            try:
                print(f"Scrapping {year}-{ctr}")
                url = get_colnect_url(year, ctr)
                items = scrap_page(url)
                df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)

            except Exception as e:
                print(e)
                break

    with open("init_state.json", "w") as file:
        json.dump({"year": year, "page": ctr}, file)

    driver.quit()

    df.to_csv("stanley_gibbons_colnect.csv", mode="a", header=False, index=False)

    print("Scrapper finished.")


if __name__ == "__main__":
    main()
