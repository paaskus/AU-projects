from flask import Flask, jsonify, render_template, request, Response
import data_crunching.simple_general_analysis as sga
import json
import os
from pathlib import Path

app = Flask(__name__, static_folder='static', static_url_path='')

@app.route('/get-result', methods=['POST'])
def get_result():
    filters = json.dumps(request.get_json())
    direcory_path = os.path.dirname(os.path.realpath(__file__))
    filename = "sample_data_5000.tsv"
    path_to_data = f'{direcory_path}/../data/{filename}'
    result = sga.crunch_the_data(path_to_data, filters)
    return Response(result, mimetype='application/json')

@app.route('/get-raw-result', methods=['GET'])
def get_raw_result():
    return None

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
