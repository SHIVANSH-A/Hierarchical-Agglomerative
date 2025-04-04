from flask import Flask, render_template
from clustering import run_clustering

app = Flask(__name__)

@app.route("/")
def index():
    run_clustering()  # Run clustering algorithm
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
