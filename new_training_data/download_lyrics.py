import time

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

options = Options()
# options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

with open("list_of_Travis_urls.txt", "r", encoding="utf-8") as file:
    links = file.readlines()

counter = 1
for link in links[:3]:
    print(link)
    try:
        driver.get(link)
        time.sleep(3)

        element = WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.TAG_NAME, 'h1'))
        )
        title = driver.find_element(By.TAG_NAME, 'h1').text

        print(f'{counter}. {title}')
        if 'Tracklist' in title:
            continue
        
        lyrics_elements = driver.find_elements(By.XPATH, "//*[starts-with(@class, 'Lyrics__Container-')]")
        lyrics = "\n\nNew verse:\n".join([lyrics_element.text for lyrics_element in lyrics_elements])

        with open(f"lyrics/{title}.txt", "w", encoding="utf-8") as file:
            file.write(lyrics)

    except Exception as e:
        print(e)

    counter += 1