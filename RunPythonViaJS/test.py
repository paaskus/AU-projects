from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route('/process', methods=['GET', 'POST'])
def processInput():
    # Get input data
    dataToProcess = request.form["param"].split(",")
    
    # Do dataprocessing...
    # example:

    result  = [dataToProcess[i]+str(i) for i in range(0,len(dataToProcess))]
    
    # Send data back result
    return jsonify(response=result)


if __name__ == '__main__':
    app.run()
