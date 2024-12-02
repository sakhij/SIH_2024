import os
from flask import Flask, render_template, request
import cohere
import pandas as pd
from scipy.spatial import KDTree
import numpy as np
from sklearn.preprocessing import LabelEncoder
import requests

app = Flask(__name__)

# Initialize Cohere with your API key
cohere_api_key = "Zppye9OdNDcXgkNIhaAVlvbFNzBnDmX6A095XOJK"
co = cohere.Client(cohere_api_key)

# Function to search for a nested key in JSON
def find_nested_key(json_obj, search_key):
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if key == search_key:
                return value
            elif isinstance(value, (dict, list)):
                result = find_nested_key(value, search_key)
                if result is not None:
                    return result
    elif isinstance(json_obj, list):
        for item in json_obj:
            result = find_nested_key(item, search_key)
            if result is not None:
                return result
    return None

# Function to find the absolute magnitude from the 'phys_par' section of the API response
def get_absolute_magnitude(data, keys, values):
    if keys in data:
        for item in data[keys]:
            if 'title' in item and item['title'].lower() == values:
                return item.get('value', 'N/A')
    return 'N/A'

# Function to extract and save orbital elements into a dictionary
def extract_orbital_elements(data):
    elements = data.get('orbit', {}).get('elements', [])
    extracted_elements = {}
    for element in elements:
        label = element.get("title")
        value = element.get("value")
        if label and value:
            extracted_elements[label] = value
    return extracted_elements

# Helper function to generate responses from the Cohere model
def generate_response(prompt, max_tokens):
    try:
        response = co.generate(
            model="command-r-plus",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.generations[0].text.strip()
    except Exception as e:
        print("Error:", e)
        return "Unable to generate a response at this time."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    absolute_magnitude = request.form["absolute_magnitude"]
    albedo = request.form["albedo"]
    eccentricity = request.form["eccentricity"]
    aphelion_distance = request.form["aphelion_distance"]

    # Load your dataset for KDTree search
    csv_file = "updated_dataset_main.csv"  # Replace with the path to your dataset
    df = pd.read_csv(csv_file)

    columns_to_search = ["albedo", "H", "e", "ad", "main_class"]
    spk_id_column = "spkid"

    if not all(col in df.columns for col in columns_to_search + [spk_id_column]):
        raise ValueError("Dataset does not contain the required columns.")

    label_encoder = LabelEncoder()
    df['main_class'] = label_encoder.fit_transform(df['main_class'])

    values = df[columns_to_search].drop(columns=['main_class']).values
    tree = KDTree(values)

    user_input = [float(albedo), float(absolute_magnitude), float(eccentricity), float(aphelion_distance), 'X-type']
    user_input_numeric = user_input[:4]

    distance, index = tree.query(user_input_numeric)
    closest_match = df.iloc[index]
    decoded_class = label_encoder.inverse_transform([closest_match['main_class']])[0]
    spk_id = closest_match[spk_id_column]

    # API integration
    api_url = f"https://ssd-api.jpl.nasa.gov/sbdb.api?spk={spk_id}&phys-par=1"
    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()
        if data:
            name = find_nested_key(data, 'fullname')
            short_name = find_nested_key(data, 'short name')
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

            # Prepare asteroid details for rendering
            asteroid_details = {
                "Name": name or 'N/A',
                "ShortName": short_name or 'N/A',
                "SPK-ID": spk_id,
                "OrbitClass": orbit_class or 'N/A',
                "pha": pha or 'N/A',
                "OrbitID": orbit_id or 'N/A',
                "H": abs_magnitude or 'N/A',
                "G": magnitude_slope or 'N/A',
                "dia": effective_diameter or 'N/A',
                "Dimensions": dimensions or 'N/A',
                "RotationPeriod": rotation_period or 'N/A',
                "Albedo": geometric_albedo or 'N/A',
                "BulkDensity": bulk_density or 'N/A',
                "a": orbital_elements['semi-major axis'],
                'eccentricity':orbital_elements['eccentricity'],
                'perihelion':orbital_elements['perihelion distance'],
                'aphelion':orbital_elements[ 'aphelion distance'],
                'inclination':orbital_elements['inclination; angle with respect to x-y ecliptic plane'],
                'longi':orbital_elements['longitude of the ascending node'],
                'args':orbital_elements['argument of perihelion']
            }
            print(asteroid_details)
            # Generate mining instructions
            mining_prompt = (
                 f"Provide a detailed step-by-step guide for safely traveling to and mining an asteroid based on the following details:\n"
                 f"{asteroid_details}\n"
                 "Provide short and precise steps"
            )
            mining_instructions_text = generate_response(mining_prompt, max_tokens=3)

            # Generate mining recommendation
            recommendation_prompt = (
                f"Based on the provided asteroid details, recommend the best methods and technologies for mining:\n"
                f"{asteroid_details}\n"
                "Suggest the best technologies and methods for efficient mining."
            )
            mining_recommendation_text = generate_response(recommendation_prompt, max_tokens=3)

            return render_template(
                "index.html",
                asteroid_details=asteroid_details,
                mining_instructions=mining_instructions_text,
                mining_recommendation=mining_recommendation_text,
                decoded_class=decoded_class
            )
        else:
            return render_template("index.html", error_message="No data found for the given SPK-ID.")
    else:
        return render_template("index.html", error_message="Failed to fetch data from NASA API. HTTP Status Code: " + str(response.status_code))

if __name__ == "__main__":
    app.run(debug=True)