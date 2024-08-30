from langdetect import detect
from tqdm import tqdm

with open("Travis_lyrics.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

with open("Travis_lyrics_cleaned.txt", "w", encoding="utf-8") as file:
    lines_set = set()
    for line in tqdm(lines):
        if line in lines_set:
            continue
        lines_set.add(line)

        try:
            # detect language
            lang = detect(line)
            if lang != 'en':
                continue
            if line[0] == "-":
                continue
            # # check if there are any characters with accent marks
            # if not line.isascii():
            #     continue
            
            if line.strip():
                line = line.replace("&#x2028;", "\n")
                file.write(line)
        except:
            continue

        

