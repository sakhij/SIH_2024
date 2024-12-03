from flask import Flask, render_template, request, jsonify
import json

app = Flask(__name__)

# Load the data from the JSON file
with open('asteroid_compositions_from_ecocell.json') as f:
    data = json.load(f)

@app.route('/')
def index():
    return render_template('graph.html')

@app.route('/get-composition', methods=['POST'])
def get_composition():
    spk_id = request.json.get('spk_id')
    
    if spk_id in data:
        composition = data[spk_id]
        return jsonify({"status": "success", "data": composition})
    else:
        return jsonify({"status": "error", "message": "SPK ID not found"})

if __name__ == "__main__":
    app.run(debug=True)