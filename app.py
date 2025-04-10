from flask import Flask, render_template, request, redirect, url_for
from clustering import run_clustering
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            linkage_matrix, steps = run_clustering(filepath)
            return render_template("result.html", steps=steps)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
