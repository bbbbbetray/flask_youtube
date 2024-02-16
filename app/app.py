from flask import Flask, request, render_template
import numpy as np
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid
# from sklearn.linear_model import LinearRegression
app = Flask(__name__)

def floats_string_to_np_arr(floats_str):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False

    floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
    return floats.reshape(len(floats), 1)

print(floats_string_to_np_arr('1,3,5'))

def make_picture(training_data_file, model, new_inp_np_arr, output_file):
    data = pd.read_pickle('AgesAndHeights.pkl')
    ages = data['Age']
    data = data[ages > 0]
    ages = data['Age']
    heights = data['Height']
    x_new = np.array(list(range(19))).reshape(19, 1)
    preds = model.predict(x_new)

    fig = px.scatter(x=ages, y=heights, title="Height vs Age of People", labels={'x': 'Age (years)',
                                                                                 'y': 'Height (inches)'})
    fig.add_trace(go.Scatter(x=x_new.reshape(19), y=preds, mode="lines", name="Model"))
    new_preds = model.predict(new_inp_np_arr)
    fig.add_trace(
        go.Scatter(x=new_inp_np_arr.reshape(len(new_inp_np_arr)), y=new_preds, name="New outputs", mode="markers",
                   marker=dict(color="purple", size=20, line=dict(color="purple", width=2))))
    #
    fig.write_image(output_file, width=800,engine="kaleido")
    fig.show()

@app.route("/", methods=["POST", "GET"])
def hello_word():
    if request.method == "GET":
        return render_template('index.html', href="static/base.svg")

    else:
        text = request.form['text']
        # return text
        random_string = uuid.uuid4().hex
        path = "static/" + random_string + ".svg"
        model = load('model.joblib')
        np_arr = floats_string_to_np_arr(text)

        make_picture('AgesAndHeights.pkl', model, np_arr, path)

        return render_template('index.html', href=path)

