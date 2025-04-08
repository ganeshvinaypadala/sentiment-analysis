from flask import Flask, request, render_template
from sentiment_model import classify_text  # Correct import

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    doctor_name = request.form.get("text", "")
    feedback = request.form.get("feedback", "")
    
    # Classify the feedback
    sentiment = classify_text(feedback)
    
    return render_template("index.html", sentiment=sentiment, doctor_name=doctor_name)

if __name__ == "__main__":
    app.run(debug=True)

