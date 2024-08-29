import requests
from bs4 import BeautifulSoup
import json

BASE_URL = "https://3d-asteroids.space/asteroids/"

def extract_asteroid_details(asteroid_url):
    response = requests.get(asteroid_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    full_name = soup.find('h1').text.strip()  
    asteroid_id = full_name.split()[0]  
    asteroid_name = ' '.join(full_name.split()[1:])  

    physical_characteristics = {
        'Mean diameter': None,
        'Rotation period': None,
        'Rotation period type': None,  
        'Albedo': None
    }

    tables = soup.find_all('table')
    if len(tables) < 2:
        print(f"Warning: Less than 2 tables found on the page {asteroid_url}")
        return {
            'id': asteroid_id,
            'full_name': full_name,
            'name': asteroid_name,
            'url': asteroid_url,
            'physical_characteristics': physical_characteristics
        }

    physical_data = tables[1].find_all('tr')
    for row in physical_data:
        cells = row.find_all('td')
        if len(cells) == 2:
            key = cells[0].text.strip()
            value = cells[1].text.strip()
            
            if key == 'Mean diameter':
                physical_characteristics['Mean diameter'] = value
            elif 'Rotation period' in key:
                physical_characteristics['Rotation period'] = value
                physical_characteristics['Rotation period type'] = key  # Store the type of rotation period
            elif key == 'Albedo':
                physical_characteristics['Albedo'] = value

    return {
        'id': asteroid_id,
        'full_name': full_name,
        'name': asteroid_name,
        'url': asteroid_url,
        'physical_characteristics': physical_characteristics
    }

def scrape_asteroid_links():
    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.text, 'html.parser')
    asteroid_links = []

    for link in soup.find_all('a', href=True):
        if "asteroids" in link['href']:
            asteroid_links.append(BASE_URL + link['href'].split("/")[-1])
    
    return asteroid_links

def main():
    asteroid_links = scrape_asteroid_links()
    all_asteroid_data = []

    for link in asteroid_links:
        print(f"Processing {link}")
        asteroid_data = extract_asteroid_details(link)
        all_asteroid_data.append(asteroid_data)

    with open('asteroid_data.json', 'w') as f:
        json.dump(all_asteroid_data, f, indent=4)

    print("Data scraping complete. Check the 'asteroid_data.json' file for output.")

if __name__ == "__main__":
    main()
