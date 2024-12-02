import os
from flask import Flask, render_template, request
import cohere

app = Flask(__name__)

# Initialize Cohere with your API key
cohere_api_key = "Zppye9OdNDcXgkNIhaAVlvbFNzBnDmX6A095XOJK"
co = cohere.Client(cohere_api_key)

# Helper function to parse and format the response into a dictionary
def parse_response(response):
    details = {}
    if response:
        sections = response.split("\n")
        for section in sections:
            if ":" in section:
                key, value = section.split(":", 1)
                details[key.strip()] = value.strip()
    return details

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

    try:
        # Generate asteroid details
        details_prompt = (
            f"Predict the asteroid's characteristics based on the following inputs:\n"
            f"Absolute Magnitude: {absolute_magnitude}\n"
            f"Albedo: {albedo}\n"
            f"Eccentricity: {eccentricity}\n"
            f"Aphelion Distance: {aphelion_distance}\n"
            f"Generate the following asteroid details:\n"
            "1. Asteroid Name\n"
            "2. Type\n"
            "3. Compositions in detail with metal names\n"
            "4. Approximate size\n"
            "5. Distance\n"
            "6. Location\n"
            "7. Speed\n"
            "8. Period of rotation\n"
            "9. Orbital period & path\n"
            "10. Eccentricity\n"
            "11. Perihelion\n"
            "12. Aphelion\n"
            "13. Inclination\n"
            "14. Close approach"
        )
        asteroid_details_text = generate_response(details_prompt, max_tokens=0)
        asteroid_details = parse_response(asteroid_details_text)

        # Generate mining instructions
        mining_prompt = (
            f"Provide a detailed step-by-step guide for safely traveling to and mining an asteroid based on the following details:\n"
            f"{asteroid_details_text}\n"
            "Provide short and precise steps."
        )
        mining_instructions_text = generate_response(mining_prompt, max_tokens=300)

        # Generate mining recommendation
        recommendation_prompt = (
            f"Based on the provided asteroid details, recommend the best methods and technologies for mining:\n"
            f"{asteroid_details_text}\n"
            "Suggest the best technologies and methods for efficient mining."
        )
        mining_recommendation_text = generate_response(recommendation_prompt, max_tokens=300)

        return render_template(
            "index.html",
            asteroid_details=asteroid_details,
            mining_instructions=mining_instructions_text,
            mining_recommendation=mining_recommendation_text,
        )

    except Exception as e:
        return render_template("index.html", error_message="An error occurred: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
