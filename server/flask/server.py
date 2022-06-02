import random
import os
from flask import Flask, request, jsonify
from keyword_spotting_service import Keyword_Spotting_Service

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():

    # get audio file and save it
    audio_file = request.files['file']
    file_name = str(random.randint(0, 100000))
    audio_file.save(file_name)

    # invoke keyword spotting service
    kss = Keyword_Spotting_Service()
    
    # predict
    predicted = kss.predict(file_name)

    # remove audio
    os.remove(file_name)

    # send back prediction
    data = {'keyword': predicted}

    return jsonify(data)

if __name__ =='__main__':
    app.run(debug=False)