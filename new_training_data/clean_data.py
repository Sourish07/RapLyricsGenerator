from langdetect import detect
import langid

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
            line = line.strip()

            guess_one = langid.classify(line)[0]
            guess_two = detect(line)
            # detect language
            if guess_one != 'en' and guess_two != 'en':
                print(line, guess_one, guess_two)
                continue

            if line[0] == "-":
                continue
            
            # # check if there are any characters with accent marks
            # if not line.isascii():
            #     continue
            
            line = line.replace("&#x2028;", "\n")
            file.write(line + "\n")
        except Exception as e:
            print(e)
            continue

        

