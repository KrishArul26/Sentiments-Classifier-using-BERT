from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from predictionFile import Prediction

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    data = request.json['data']

    try:
        if len(data) == 1:
            data.append("")
            predictorObj = Prediction()
            result = predictorObj.executeProcessing(data)
            result.pop()
            return jsonify({"Results": str(result)})
        else:
            predictorObj = Prediction()
            result = predictorObj.executeProcessing(data)
            return jsonify({"Results": str(result)})
    except:
        return {"Results": "Wrong Data Format Sent"}


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

