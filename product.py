import os
from flask import Flask, render_template, request,jsonify
import cohere
import pandas as pd
from scipy.spatial import KDTree
import numpy as np
from sklearn.preprocessing import LabelEncoder
import requests
import json
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Initialize Cohere with your API key
cohere_api_key = "Zppye9OdNDcXgkNIhaAVlvbFNzBnDmX6A095XOJK"
#cohere_api_key = "bWAmXD1sME2U2Tf4Ulvcp4MpQVpWpDjFENy9e3VN"
co = cohere.Client(cohere_api_key)

with open('asteroid_compositions_from_ecocell.json') as f:
    compo = json.load(f)

csv_file = "updated_dataset_main.csv"  # Replace with the path to your dataset
# Load your dataset for KDTree search
df = pd.read_csv(csv_file)

global decoded_class
global asteroid_details

# Helper function to format responses for better readability
def format_response(response):
    sections = response.split("\n")
    formatted = ""
    for section in sections:
        if ":" in section:
            formatted += f"<b>{section.split(':')[0]}:</b> {section.split(':')[1]}<br>"
        else:
            formatted += f"{section}<br>"
    return formatted.strip()


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

@app.route('/get-composition', methods=['POST'])
def get_composition():
    global decoded_class
    spk_id = request.json.get('spk_id')
    print(spk_id in compo)
    
    if spk_id in compo:
        composition = compo[spk_id]
        print(f"Composition for SPK ID {spk_id}:")
        print(composition)
        return jsonify({"status": "success", "data": composition})
    else:
        if decoded_class=='S-type':
            composition = compo["20000433"]
            print(f"Composition for SPK ID {spk_id}:")
            print(composition)
            return jsonify({"status": "success", "data": composition})
        elif decoded_class=='C-type':
            composition = compo["20001580"]
            print(f"Composition for SPK ID {spk_id}:")
            print(composition)
            return jsonify({"status": "success", "data": composition})
        elif decoded_class=='X-type':
            composition = compo["20005751"]
            print(f"Composition for SPK ID {spk_id}:")
            print(composition)
            return jsonify({"status": "success", "data": composition})

@app.route("/analyze", methods=["POST"])
def analyze():
    global decoded_class
    global asteroid_details
    global mission_cost
    global benefits

    # Load the pre-trained CNN model
    cnn_model = load_model('cnn_model_new_data.h5')  # Replace with your actual CNN model file path

    absolute_magnitude = request.form["absolute_magnitude"]
    albedo = request.form["albedo"]
    eccentricity = request.form["eccentricity"]
    aphelion_distance = request.form["aphelion_distance"]

    columns_to_search = ["albedo", "H", "e", "ad", "main_class"]
    spk_id_column = "spkid"

    if not all(col in df.columns for col in columns_to_search + [spk_id_column]):
        raise ValueError("Dataset does not contain the required columns.")

    # Prepare input data for the CNN model
    user_input = [float(albedo), float(absolute_magnitude), float(eccentricity), float(aphelion_distance)]
    cnn_input = np.array(user_input).reshape(1, -1)  # Reshape to match the expected input of the CNN model

    # Predict the class using the CNN model
    cnn_predicted_class_index = cnn_model.predict(cnn_input)  # Predict the class index using the CNN model
    cnn_predicted_class_index = np.argmax(cnn_predicted_class_index, axis=1)[0]  # Extract the predicted class index

    # Map the CNN predicted index to the actual class using label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(df['main_class'])  # Fit encoder on existing classes
    decoded_class = label_encoder.inverse_transform([int(cnn_predicted_class_index)])[0]

    # Print the predicted class from CNN model
    print(f"Predicted Class (CNN): {decoded_class}")

    values = df[columns_to_search].drop(columns=['main_class']).values
    tree = KDTree(values)

    user_input = [float(albedo), float(absolute_magnitude), float(eccentricity), float(aphelion_distance), 'X-type']
    user_input_numeric = user_input[:4]

    distance, index = tree.query(user_input_numeric)
    closest_match = df.iloc[index]
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
                "SpectralClass": decoded_class,
                "SPKID": spk_id,
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

            print("\nMission Cost Estimation:")

            # Mission Design and Planning Cost
            f_orbit = 1 + float(asteroid_details["eccentricity"]) + (float(asteroid_details["inclination"]) / 10)
            design_cost = 500 * f_orbit  # $500 million base
            effective_diameter=asteroid_details["dia"]
            bulk_density=asteroid_details["BulkDensity"]
            geometric_albedo=asteroid_details["Albedo"]
            

            if effective_diameter == "N/A":
                effective_diameter = 100.0  # Assume a default effective diameter in km
            else:
                effective_diameter = float(effective_diameter)
            
            # Spacecraft Development and Launch Cost
            dev_cost = float(effective_diameter) * 10  # $10 million per km
            launch_cost = 3_000  # Assumed $3 billion for simplicity

            if bulk_density != "N/A":
                bulk_density = float(bulk_density)
            else:
                if decoded_class == "C-type":
                    bulk_density = 1.3  # kg/m^3 (carbonaceous)
                elif decoded_class == "S-type":
                    bulk_density = 2.7  # kg/m^3 (silicate-rich)
                elif decoded_class == "X-type":
                    bulk_density = 7    # kg/m^3 (metallic)
                else:
                    bulk_density = 2.7  # Default to S-type

            # Asteroid Capture and Processing Cost
            capture_cost = 2_000  # Assumed $2 billion for capture
            process_cost = float(bulk_density) * (1 / float(geometric_albedo)) * 10  # $10 million per unit

            total_cost = design_cost + dev_cost + launch_cost + capture_cost + process_cost

            print(f"Mission Design and Planning Cost: ${design_cost:.2f} million")
            print(f"Spacecraft Development and Launch Cost: ${dev_cost + launch_cost:.2f} million")
            print(f"Asteroid Capture and Processing Cost: ${capture_cost + process_cost:.2f} million")
            print(f"Total Mission Cost: ${total_cost:.2f} million")

            # Estimated Benefits Calculation
            print("\nEstimated Benefits:")

            radius = (float(effective_diameter) / 2) * 1e3  # Convert effective diameter to radius in meters
            volume_mined = (4/3) * np.pi * (radius**3)  # Volume in cubic meters

            # Calculate mass of asteroid
            mass_of_asteroid = bulk_density * volume_mined  # Mass in kg

            # Water Ice (5% of mass)
            water_ice_mass = 0.05 * mass_of_asteroid  # in kg
            water_ice_value_range = (water_ice_mass * 10**-9 * 10, water_ice_mass * 10**-9 * 30)  # in billion USD
            print(f"Water Ice: {water_ice_mass * 10**-9} million tons (valued at ${water_ice_value_range[0]} - ${water_ice_value_range[1]} billion)")

            # Precious Metals (10% of mass)
            precious_metals_mass = 0.1 * mass_of_asteroid  # in kg
            precious_metals_value_range = (precious_metals_mass * 10**-6 * 5, precious_metals_mass * 10**-6 * 15)  # in billion USD
            print(f"Precious metals: {precious_metals_mass * 10**-6} tons (valued at ${precious_metals_value_range[0]} - ${precious_metals_value_range[1]} billion)")

            # Organic Compounds (1% of mass)
            organic_compounds_mass = 0.01 * mass_of_asteroid  # in kg
            organic_compounds_value_range = (organic_compounds_mass * 10**-9 * 1, organic_compounds_mass * 10**-9 * 3)  # in billion USD
            print(f"Organic Compounds: {organic_compounds_mass * 10**-9} million tons (valued at ${organic_compounds_value_range[0]} - ${organic_compounds_value_range[1]} billion)")

            # Total Value Range
            total_value_range = (
                water_ice_value_range[0] + precious_metals_value_range[0] + organic_compounds_value_range[0],
                water_ice_value_range[1] + precious_metals_value_range[1] + organic_compounds_value_range[1]
            )
            print(f"Total estimated value: ${total_value_range[0]} - ${total_value_range[1]} billion")

            total_value_range_quadrillion = (total_value_range[0] / 1_000_000, total_value_range[1] / 1_000_000)
            print(f"Total estimated value in quad: ${total_value_range_quadrillion[0]} - ${total_value_range_quadrillion[1]} quadrillion")

            if total_value_range[0] < 100_000_000:
                total_value_display = {
                    "value_range": total_value_range,
                    "unit": "billion"  # Display in billions
                }
            else:
                total_value_display = {
                    "value_range": total_value_range_quadrillion,
                    "unit": "quadrillion"  # Display in billions
                }

            mission_cost={
                        "design_cost": design_cost,
                        "dev_cost": dev_cost,
                        "launch_cost": launch_cost,
                        "capture_cost": capture_cost,
                        "process_cost": process_cost,
                        "total_cost": total_cost
                    }
            
            benefits={
                        "water_ice": {
                            "mass": water_ice_mass * 10**-9,
                            "value_range": water_ice_value_range
                        },
                        "precious_metals": {
                            "mass": precious_metals_mass * 10**-6,
                            "value_range": precious_metals_value_range
                        },
                        "organic_compounds": {
                            "mass": organic_compounds_mass * 10**-9,
                            "value_range": organic_compounds_value_range
                        },
                       "total_value_display": total_value_display
                    }

            # Generate mining instructions
            mining_prompt = (
                 f"Provide a detailed step-by-step guide for safely traveling to and mining an asteroid based on the following details:\n"
                 f"{asteroid_details}\n"
                 "Provide short and precise steps"
            )
            mining_instructions_text = format_response(generate_response(mining_prompt, max_tokens=300))

            # Generate mining recommendation
            recommendation_prompt = (
                f"Based on the provided asteroid details, recommend the best methods and technologies for mining:\n"
                f"{asteroid_details}\n"
                "Suggest the best technologies and methods for efficient mining."
            )
            mining_recommendation_text = format_response(generate_response(recommendation_prompt, max_tokens=300))
            return render_template(
                "index.html",
                asteroid_details=asteroid_details,
                mining_instructions=mining_instructions_text,
                mining_recommendation=mining_recommendation_text,
                mission_cost=mission_cost,
                benefits= benefits,
                decoded_class=decoded_class, 
                spk_id=asteroid_details['SPKID']
            )
        else:
            return render_template("index.html", error_message="No data found for the given SPK-ID.")
    else:
        return render_template("index.html", error_message="Failed to fetch data from NASA API. HTTP Status Code: " + str(response.status_code))

@app.route("/get-mission-data", methods=["POST"])
def get_mission_data():
    print('hi')
    global asteroid_details
    global mission_cost
    global benefits

    return jsonify({
        "status": "success",
        "data": {
            "mission_cost": mission_cost,
            "benefits": benefits
        }
    })

# @app.route('/nasa/<path:path>')
# def proxy(path):
#     # Construct the NASA URL
#     nasa_url = f'https://ssd.jpl.nasa.gov/{path}'
    
#     # Forward the request to NASA
#     resp = requests.get(nasa_url, 
#                         headers={k: v for k, v in request.headers if k.lower() != 'host'},
#                         stream=True)
    
#     # Create response object
#     excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection',
#                         'x-frame-options']  # Remove x-frame-options to allow embedding
#     headers = [(name, value) for (name, value) in resp.raw.headers.items()
#                if name.lower() not in excluded_headers]
    
#     # Add CORS headers
#     headers.append(('Access-Control-Allow-Origin', '*'))
    
#     return Response(resp.content, resp.status_code, headers)

if __name__ == "__main__":
    app.run(debug=True)