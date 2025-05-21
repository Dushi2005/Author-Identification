from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load("author_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        text = request.form["text"]
        features = vectorizer.transform([text])
        prediction = model.predict(features)[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
