import re

from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# ---------------------------------------------------------------
# Training data (first 10 rows are the ones from the lab record)
# ---------------------------------------------------------------
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
    # extra rows so the model generalises a little better
    "Urgent claim your cash reward today",
    "You have been selected as a lucky winner",
    "Limited time offer get free money now",
    "Hurry click the link to claim your prize",
    "Congratulations winner you won a cash prize of 50000",
    "Good morning please bring the project report for the meeting",
    "The meeting is scheduled for tomorrow morning",
    "Please review the attached document and share feedback",
    "Thanks for your help with the assignment",
    "Reminder your class starts at ten tomorrow",
]
labels = ["spam"] * 5 + ["ham"] * 5 + ["spam"] * 5 + ["ham"] * 5

# TF-IDF + Naive Bayes
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
X = vectorizer.fit_transform(messages)
model = MultinomialNB()
model.fit(X, labels)

SPAM_WORDS = ["free", "winner", "won", "prize", "reward", "cash",
              "lottery", "offer", "claim", "money", "selected", "lucky"]
URGENCY_WORDS = ["urgent", "now", "today", "immediately", "hurry",
                 "limited time", "expires", "last chance"]
LINK_PATTERN = re.compile(r"(https?://|www\.|click here|click the link)", re.I)


def count_words(text, words):
    """Count whole-word matches (so 'know' does not match 'now')."""
    return sum(
        1 for w in words
        if re.search(r"\b" + re.escape(w) + r"\b", text)
    )


def analyze_message(message, prediction):
    text = message.lower()
    spam_count = count_words(text, SPAM_WORDS)
    urgency_count = count_words(text, URGENCY_WORDS)
    has_link = bool(LINK_PATTERN.search(text))

    if prediction == "spam":
        if spam_count > 0 and urgency_count > 0:
            reason = "Contains promotional and urgent words."
        elif spam_count > 0:
            reason = "Contains promotional or prize-related words."
        elif urgency_count > 0:
            reason = "Contains urgent timing language."
        else:
            reason = "Similar to previously detected spam messages."
    else:
        reason = ("The message does not contain strong spam indicators such "
                  "as prize claims, suspicious offers, or urgent action requests.")

    breakdown = {
        "timing": ("High Urgency", True) if urgency_count else ("Normal", False),
        "content": ("Suspicious", True) if spam_count else ("Normal", False),
        "pattern": ("Suspicious Pattern", True)
        if (prediction == "spam" and (has_link or spam_count))
        else ("No Suspicious Pattern", False),
    }
    return reason, breakdown


@app.route("/", methods=["GET", "POST"])
def home():
    result = confidence = reason = breakdown = None
    is_spam = False
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "")
        if message.strip():
            vec = vectorizer.transform([message])
            prediction = model.predict(vec)[0]
            probabilities = model.predict_proba(vec)[0]
            confidence = round(max(probabilities) * 100, 2)

            is_spam = prediction == "spam"
            result = "SPAM MAIL DETECTED" if is_spam else "SAFE EMAIL"
            reason, breakdown = analyze_message(message, prediction)

    return render_template(
        "index.html",
        result=result,
        is_spam=is_spam,
        confidence=confidence,
        message=message,
        reason=reason,
        breakdown=breakdown,
    )


if __name__ == "__main__":
    app.run(debug=True)
