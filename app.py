import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, url_for
from clustering import hierarchical_clustering, plot_dendrogram

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            df = pd.read_csv(filepath)
            data = df.select_dtypes(include=[np.number]).values  # Only use numerical columns
            linkage_matrix, steps, intermediate_plots = hierarchical_clustering(data)
            plot_dendrogram(linkage_matrix)
            return render_template("index.html", head=df.head().to_html(), steps=steps, 
                                   image_url=url_for('static', filename='dendrogram.png'), 
                                   intermediate_plots=intermediate_plots)
    return render_template("index.html", head=None, steps=None, image_url=None, intermediate_plots=None)

if __name__ == "__main__":
    app.run(debug=True)
