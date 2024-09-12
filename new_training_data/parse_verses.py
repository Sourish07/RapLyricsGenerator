"""
Script to iterate through all songs in lyrics folder and fetch the ones that below to Travis Scott
Viable options:
[Verse 1: Travis Scott]
[Chorus: Travis Scott]
[Post-Chorus: Travis Scott]
[Verse 1]
[Verse 2: Travis Scott & The Weeknd]
[Outro]
"""

import os
import re

def is_travis_verse(line, verse_pattern):
    """Check if a line indicates the start of a Travis Scott verse, chorus, or duet."""
    return bool(verse_pattern.match(line))

def write_verses_to_file(verses, output_file):
    """Write the unique verses to the output file."""
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for verse in verses:
            outfile.write(verse + '\n\n')  # Add a blank line between verses

def extract_travis_verses(directory):
    # Regex to capture Travis Scott's verses, duets, choruses, and post-choruses
    verse_pattern = re.compile(
        r'.*?: Travis Scott(?: & [^\]]+)?\]|Verse \d+\]|Chorus\]|Post-Chorus\]', re.IGNORECASE
    )

    # Set to store unique verses
    unique_verses = set()

    # Iterate through all text files in the directory
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith('.txt'):
                continue

            file_path = entry.path
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                chunks = text.split("[")
                for chunk in chunks:
                    chunk_lines = chunk.split("\n")
                    if not re.match(verse_pattern, chunk_lines[0]):
                        continue

                    cleaned_chunk_lines = [l for l in chunk_lines[1:] if len(l) > 0]

                    if len(cleaned_chunk_lines) <= 2:
                        continue
                    
                    final_verse = "\n".join(cleaned_chunk_lines)
                    unique_verses.add(final_verse)

    # Write all unique verses to a new file
    output_file = os.path.join('travis_scott_verses.txt')
    write_verses_to_file(unique_verses, output_file)

    print(f"Extracted {len(unique_verses)} unique Travis Scott verses to {output_file}")


if __name__ == "__main__":
    extract_travis_verses("lyrics")
