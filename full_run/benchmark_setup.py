import requests

def fetch_text_from_gutenberg(book_id):
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    response = requests.get(url)
    
    if response.status_code == 200:
        response.encoding = 'utf-8'  # Ensure the correct encoding is used
        return response.text
    else:
        return None

def save_text_to_file(text, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

def main():
    book_id = 174  # Project Gutenberg ID for "Frankenstein"
    filename = "/Users/brandon/Desktop/MonteIntelligence/corpora/the_picture_of_dorian_gray.txt"
    
    print("Fetching text from Project Gutenberg...")
    text = fetch_text_from_gutenberg(book_id)
    
    if text:
        print("Saving text to file...")
        save_text_to_file(text, filename)
        print(f"Text saved to {filename}")
    else:
        print("Failed to fetch text.")

if __name__ == "__main__":
    main()
