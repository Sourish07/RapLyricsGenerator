import time
import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.firefox import GeckoDriverManager

options = Options()
#options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=options)

with open("list_of_Travis_urls.txt", "r", encoding="utf-8") as file:
    links = file.readlines()

counter = 1
for link in links:
    print(link)
    try:
        driver.get(link)
        time.sleep(3)

        element = WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.TAG_NAME, 'h1'))
        )
        title = driver.find_element(By.TAG_NAME, 'h1').text

        if os.path.isfile(f"lyrics/{title}.txt"):
            continue

        print(f'{counter}. {title}')
        if 'Tracklist' in title:
            continue
        
        lyrics_elements = driver.find_elements(By.XPATH, "//*[starts-with(@class, 'Lyrics__Container-')]")
        lyrics = "\n".join([lyrics_element.text for lyrics_element in lyrics_elements])

        with open(f"lyrics/{title}.txt", "w", encoding="utf-8") as file:
            file.write(lyrics)

    except Exception as e:
        print(e)

    counter += 1
