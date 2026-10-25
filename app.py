import re
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Training data
messages = [
    "Congratulations you won a free lottery",
    "Claim your free reward now",
    "Win a free mobile phone",
    "You won a cash prize",
    "Click here to get your prize",
    "Please send me the project report",
    "Can we meet tomorrow",
    "Your appointment is confirmed",
    "Please submit the assignment",
    "Thank you for your support",
]
labels = ["spam"] * 5 + ["ham"] * 5

# TF-IDF + Naive Bayes model
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(messages)
model = MultinomialNB()
model.fit(X, labels)

spam_words = ["free", "winner", "won", "prize", "reward", "cash",
              "lottery", "offer", "claim", "money"]
urgency_words = ["urgent", "now", "today", "immediately", "hurry"]


@app.route("/", methods=["GET", "POST"])
def home():
    message = ""
    result = confidence = reason = None

    if request.method == "POST":
        message = request.form.get("message", "")

        if message.strip():
            vector = vectorizer.transform([message])
            prediction = model.predict(vector)[0]
            confidence = round(max(model.predict_proba(vector)[0]) * 100, 2)

            words = re.findall(r"\w+", message.lower())
            has_spam = any(w in words for w in spam_words)
            has_urgency = any(w in words for w in urgency_words)

            if prediction == "ham":
                result = "SAFE EMAIL"
                reason = "No strong spam indicators were detected."
            else:
                result = "SPAM MAIL DETECTED"
                if has_spam and has_urgency:
                    reason = "Contains promotional and urgent words."
                elif has_spam:
                    reason = "Contains promotional or prize-related words."
                elif has_urgency:
                    reason = "Contains urgent timing language."
                else:
                    reason = "Similar to previously detected spam messages."

    return render_template("index.html", message=message, result=result,
                           confidence=confidence, reason=reason)


if __name__ == "__main__":
    app.run(debug=True)
