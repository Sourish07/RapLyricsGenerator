from bs4 import BeautifulSoup
import requests

from selenium import webdriver as web
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.wait import WebDriverWait


def find_lyrics(url, artist_name):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    # lyrics = soup.find('div', class_='lyrics').get_text()
    lyrics = driver.find_element_by_class_name("lyrics").text

    # Text processing
    split_lyrics = lyrics.split("\n")
    final_lyrics = ""
    artist_first_name = artist_name.split(" ")[0]
    to_append = False
    for line in split_lyrics:
        if '[' in line and ':' in line:
            temp = line.split(":")[1].split(" ")
            if artist_first_name.lower() in temp[1].lower():
                to_append = True
                continue
            else:
                to_append = False
                continue
        elif '[' in line:
            to_append = True
            continue
        if to_append and line != "":
            final_lyrics += line + "\n"
    return final_lyrics


PATH = 'C:\\Program Files (x86)\\chromedriver.exe'
driver = web.Chrome(PATH)
artist_name = "21 Savage"
artist_first_name = "21"
base_url = 'https://genius.com/artists/21-savage'
driver.get(base_url)

skip_urls_gathering = False

if not skip_urls_gathering:
    # Open modal
    driver.find_element_by_xpath(f'//div[normalize-space()="Show all songs by {artist_name}"]').click()
    song_locator = By.CSS_SELECTOR, 'a.mini_card.mini_card--small'
    # Wait for first XHR complete
    wait(driver, 25).until(EC.visibility_of_element_located(song_locator))
    # Get current length of songs list
    current_len = len(driver.find_elements(*song_locator))

    while True:
        # Load new XHR until it's possible
        driver.find_element(*song_locator).send_keys(Keys.END)
        try:
            wait(driver, 25).until(lambda x: len(driver.find_elements(*song_locator)) > current_len)
            current_len = len(driver.find_elements(*song_locator))
        # Return full list of songs
        except TimeoutException:
            songs_list = [song.get_attribute('href') for song in driver.find_elements(*song_locator)]
            break

    print(len(songs_list))
    with open(f"list_of_{artist_first_name}_urls.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(songs_list))


with open(f"list_of_{artist_first_name}_urls.txt", "r", encoding="utf-8") as file:
    links = file.readlines()

output = open(f'{artist_first_name}_lyrics.txt', 'w', encoding="utf-8")

counter = 1
for link in songs_list:
    try:
        driver.get(link)

        element = WebDriverWait(driver, 25).until(
            EC.presence_of_element_located((By.TAG_NAME, 'h1'))
        )
        title = driver.find_element_by_tag_name('h1').text

        print(f'{counter}. {title}')
        if 'Tracklist' in title:
            continue

        output.write(find_lyrics(link, artist_name))
        counter += 1
    except Exception as e:
        print("ERROR on previous title" + str(e))
        continue
output.close()


