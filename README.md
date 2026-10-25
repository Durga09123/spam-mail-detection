# Smart Spam Mail Detection (NLP Lab – Project 8)

Flask web app that classifies an email as **spam** or **safe** using TF-IDF + Multinomial Naive Bayes.

## Run in VS Code

```bash
cd spam-mail-detector
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Deploy on Vercel

1. Create a GitHub repo and push this folder so that `app.py` is in the repo root:
   ```bash
   git init
   git add .
   git commit -m "Spam mail detector"
   git branch -M main
   git remote add origin https://github.com/<your-username>/spam-mail-detector.git
   git push -u origin main
   ```
2. Go to https://vercel.com/new, import the repo, keep the defaults, click **Deploy**.
   (Vercel detects Flask from `app.py` + `requirements.txt`.)

## Test messages

- Spam: `Subject: You Won a ₹50,000 Cash Prize! Claim your FREE reward NOW. Click here, limited time offer.`
- Safe: `Hi Divya, Good morning. Please bring the project report for tomorrow's meeting.`
