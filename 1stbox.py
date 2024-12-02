import pandas as pd
from scipy.spatial import KDTree
import numpy as np
from sklearn.preprocessing import LabelEncoder
import requests

# Function to search for a nested key in JSON
def find_nested_key(json_obj, search_key):
    # If the json_obj is a dictionary, search through its keys
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if key == search_key:
                return value
            # If the value is another dictionary or list, recurse
            elif isinstance(value, (dict, list)):
                result = find_nested_key(value, search_key)
                if result is not None:
                    return result

    # If the json_obj is a list, search through its elements
    elif isinstance(json_obj, list):
        for item in json_obj:
            result = find_nested_key(item, search_key)
            if result is not None:
                return result

    # Return None if key is not found
    return None

# Function to find the absolute magnitude from the 'phys_par' section of the API response
def get_absolute_magnitude(data, keys, values):
    """
    This function checks the 'phys_par' list for the title 'absolute magnitude'
    and returns the corresponding 'value'.
    """
    if keys in data:
        # Iterate through the items in 'phys_par'
        for item in data[keys]:
            # Check if the title is 'absolute magnitude'
            if 'title' in item and item['title'].lower() == values:
                # If the title matches, return the 'value' key
                return item.get('value', 'N/A')
    return 'N/A'

# Function to extract and save orbital elements into a dictionary
def extract_orbital_elements(data):
    """
    Extracts orbital elements from the API response and saves them into a dictionary.
    """
    elements = data.get('orbit', {}).get('elements', [])
    extracted_elements = {}
    for element in elements:
        label = element.get("title")
        value = element.get("value")
        if label and value:  # Ensure both label and value are present
            extracted_elements[label] = value
    return extracted_elements

# Data preprocessing and KDTree search
csv_file = "updated_dataset_main.csv"  # Replace with the path to your dataset
df = pd.read_csv(csv_file)

# Columns to be used in the search
columns_to_search = ["albedo", "H", "e", "ad", "main_class"]  # Replace with the actual column names
spk_id_column = "spkid"  # Replace with the name of the SPK-ID column in your dataset

# Check if the required columns exist
if not all(col in df.columns for col in columns_to_search + [spk_id_column]):
    raise ValueError("Dataset does not contain the required columns.")

# Encode the 'main_class' column using LabelEncoder (to handle categorical data)
label_encoder = LabelEncoder()
df['main_class'] = label_encoder.fit_transform(df['main_class'])

# Build a KDTree for efficient nearest neighbor search
values = df[columns_to_search].drop(columns=['main_class']).values  # Exclude 'main_class' during KDTree build
tree = KDTree(values)

# Ask the user for input
user_input = [0.1203, 6.21, 0.1341, 3.31444, 'X-type']

# Now we perform the KDTree query using only the numeric columns (not 'main_class')
user_input_numeric = user_input[:4]  # Extract the first 4 values (excluding 'main_class')

# Find the closest match in the dataset
distance, index = tree.query(user_input_numeric)
closest_match = df.iloc[index]

# Decode the 'main_class' column back to the original label
decoded_class = label_encoder.inverse_transform([closest_match['main_class']])[0]

# Extract the SPK-ID
spk_id = closest_match[spk_id_column]

# Print closest match details
print("\nClosest Match Found:")
print(closest_match[columns_to_search])
print(f"SPK-ID: {spk_id}")
print(f"Decoded main_class: {decoded_class}")

# API integration
api_url = f"https://ssd-api.jpl.nasa.gov/sbdb.api?spk={spk_id}&phys-par=1"

# Send the request to the API
response = requests.get(api_url)

# Check if the request was successful
if response.status_code == 200:
    # Print the raw response to check its structure
    #print("\nRaw API Response:")
    data = response.json()
    #print(data)

    if data:
        # Extract and print the corresponding values from the API response
        name = find_nested_key(data, 'fullname')
        short_name = find_nested_key(data, 'shortname')
        spk_id_api = find_nested_key(data, 'spkid')
        orbit_class = find_nested_key(data, 'orbit_class')
        pha = find_nested_key(data, 'pha')
        orbit_id = find_nested_key(data, 'orbit_id')
        abs_magnitude = get_absolute_magnitude(data, 'phys_par', 'absolute magnitude')
        magnitude_slope = get_absolute_magnitude(data, 'phys_par', 'magnitude slope')
        effective_diameter = get_absolute_magnitude(data, 'phys_par', 'diameter')
        dimensions = get_absolute_magnitude(data, 'phys_par', 'extent')
        rotation_period = get_absolute_magnitude(data, 'phys_par', 'rotation period')
        geometric_albedo = get_absolute_magnitude(data, 'phys_par', 'geometric albedo')
        bulk_density = get_absolute_magnitude(data, 'phys_par', 'bulk density')

        # Extract and save orbital elements using the function
        orbital_elements = extract_orbital_elements(data)

        # Print asteroid details
        print("\nAsteroid Details from NASA API:")
        print(f"Name: {name or 'N/A'}")
        print(f"Short Name: {short_name or 'N/A'}")
        print(f"SPK-ID: {spk_id_api or 'N/A'}")
        print(f"Orbit Class: {orbit_class or 'N/A'}")
        print(f"Not a Potentially Hazardous Asteroid (PHA): {pha or 'N/A'}")
        print(f"Orbit ID: {orbit_id or 'N/A'}")
        print(f"Absolute Magnitude (H): {abs_magnitude or 'N/A'}")
        print(f"Magnitude Slope (G): {magnitude_slope or 'N/A'}")
        print(f"Effective Diameter: {effective_diameter or 'N/A'}")
        print(f"Dimensions: {dimensions or 'N/A'}")
        print(f"Rotation Period: {rotation_period or 'N/A'}")
        print(f"Geometric Albedo: {geometric_albedo or 'N/A'}")
        print(f"Bulk Density: {bulk_density or 'N/A'}")

        for label, value in orbital_elements.items():
            print(f"{label}: {value}")
    else:
        print("No data found for the given SPK-ID.")
else:
    print(f"Failed to fetch data from NASA API. HTTP Status Code: {response.status_code}")
