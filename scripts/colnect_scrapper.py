import time

import pandas as pd

from selenium import webdriver
from bs4 import BeautifulSoup

cookies = {
    "cnv2sess": "v37r2b5mau7hpquvqhvrdv93fg",
    "cnv2user2": "c6rqinp6yx44wss0s8o8ookwo8c4ss8",
    "cnv2_forum_u": "379164",
    "cnv2_forum_sid": "6f970be75a78f9f809bdaef56f2c4600",
}

driver = webdriver.Chrome()

for cookie in cookies:
    cookie_item = {
        'domain': 'colnect.com',
        'name': cookie,
        'value': cookies[cookie],
    }
    driver.execute_cdp_cmd('Network.setCookie', cookie_item)

driver.execute_cdp_cmd('Network.disable', {})

colnect_url = "https://colnect.com/en/stamps/list/country/224-United_Kingdom_of_Great_Britain_Northern_Ireland/sort/by_issue_date"
img_placeholder_url = "i.colnect.net/items/thumb/none-stamps.jpg"


def scrap_page(url):
    driver.get(url)
    time.sleep(0.4)

    current_url = driver.current_url
    if url != current_url:
        print(f"Finished scrapping the last page: {current_url}")
        return True, None

    soup = BeautifulSoup(driver.page_source, features='lxml')

    items_parent = soup.find('div', id='plist_items')

    if items_parent is None:
        print("They might have blocked us. Exiting...")
        return True, None

    # ctr = 1
    items = []
    for child in items_parent.findChildren('div', recursive=False):
        item = {}
        image_url = child.find('img')['data-src'][2:]
        if image_url == img_placeholder_url:
            continue

        item['image_url'] = 'https://i.colnect.net/b' + image_url[15:]
        item['url'] = "https://colnect.com" + child.find('a')['href']
        items.append(item)
        if not item['image_url'].startswith('i.colnect.net'):
            print(f"Image URL: {item['image_url']}")

    return False, items


def main():
    print("Starting the scrapper...")
    df = pd.DataFrame(columns=['url', 'image_url'])

    ctr = 1
    finished = False
    while not finished:
        print(f"Scrapping page {ctr}")
        finished, items = scrap_page(colnect_url if ctr == 1 else f"{colnect_url}/page/{ctr}")
        if items:
            df = pd.concat([df, pd.DataFrame(items)], ignore_index=True)
        ctr += 1

    driver.quit()

    df.to_csv('colnect.csv', index=False)
    print("Scrapper finished.")


if __name__ == "__main__":
    main()
