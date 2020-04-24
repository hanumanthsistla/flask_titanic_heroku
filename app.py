import pandas as pd
from flask import Flask, jsonify, request
import pickle

# load model

# model = pickle.load(open('model.pkl','rb'))

## New code for model - Dr.S.Hanumanth Sastry

model = pickle.load(open('/home/hduser/Desktop/BF_Tuyeres_ML_Project/models_saved_dir/deploy-mlm-flask-heroku-master/model_titanic.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    result = model.predict(data_df)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 8000, debug=True)
