from flask import Flask, jsonify, render_template, request, Response
import data_crunching.simple_general_analysis as sga
import json

app = Flask(__name__, static_folder='static', static_url_path='')

@app.route('/get-result', methods=['POST'])
def get_result():
    filters = json.dumps(request.get_json())
    path_to_data = '~/JPdatatool/JPdata/jyllandsposten_20170402-20170402_18014v2.tsv'
    result = sga.crunch_the_data(path_to_data, filters)
    return Response(result, mimetype='application/json')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
