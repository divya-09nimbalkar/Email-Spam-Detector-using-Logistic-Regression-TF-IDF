# ============================================================
# Email Spam Detector — Logistic Regression + TF-IDF
# Dataset : Synthetic email dataset (built-in, no download needed)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)

# ── 1. Sample Dataset ────────────────────────────────────────
print("=" * 55)
print("   EMAIL SPAM DETECTOR - TF-IDF + Logistic Regression")
print("=" * 55)

spam_emails = [
    "Congratulations! You won a $1000 gift card. Click here now!",
    "FREE MONEY! Claim your prize today. Limited time offer!",
    "You have been selected for a lottery. Send your details.",
    "Buy cheap meds online. No prescription needed. Huge discount!",
    "Make money fast from home. Work only 2 hours a day!",
    "URGENT: Your account will be suspended. Verify now!",
    "Get rich quick scheme revealed. Click the link below!",
    "Hot singles in your area. Meet them tonight for FREE!",
    "Win a free iPhone! Just fill out this survey immediately!",
    "Earn $5000 per week working from home. No experience needed!",
    "Special offer just for you! Discount expires in 1 hour!",
    "You are our lucky winner today. Claim prize before midnight!",
    "Lowest mortgage rates ever! Refinance your home today!",
    "Lose 30 pounds in 30 days with our miracle pill!",
    "Click to unsubscribe and win a surprise gift!",
    "Cheap Rx drugs — no prescription! Order online now.",
    "Exclusive deal: 90% off luxury watches. Click now!",
    "Your payment failed. Update your card details ASAP!",
    "Increase your income by 500%. This secret method works!",
    "Nigerian prince needs help. You will receive 20% commission.",
]

ham_emails = [
    "Hi, can we schedule a meeting for tomorrow at 3pm?",
    "Please find attached the report for Q3 analysis.",
    "The project deadline has been moved to next Friday.",
    "Thanks for your help with the code review yesterday.",
    "I will be out of office from Monday to Wednesday.",
    "Could you please send me the updated presentation slides?",
    "Let us catch up over coffee this week if you are free.",
    "The team lunch is scheduled for Thursday at noon.",
    "Please review the pull request when you get a chance.",
    "I have updated the documentation as per your feedback.",
    "Your interview is confirmed for 10am on Tuesday.",
    "The client meeting has been rescheduled to 2pm Friday.",
    "Happy birthday! Hope you have a wonderful day.",
    "Please submit your timesheet before end of day Friday.",
    "The server maintenance is scheduled for Sunday 2-4am.",
    "Can you please share the quarterly budget report?",
    "The new feature deployment went live this morning.",
    "Your subscription receipt for this month is attached.",
    "We are planning a team outing next weekend. Are you in?",
    "Please confirm your attendance for the training session.",
]

emails = spam_emails + ham_emails
labels = [1] * len(spam_emails) + [0] * len(ham_emails)

df = pd.DataFrame({"email": emails, "label": labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n Total samples : {len(df)}")
print(f" Spam emails   : {df['label'].sum()}")
print(f" Ham  emails   : {len(df) - df['label'].sum()}")

# ── 2. Text Preprocessing & Vectorization ───────────────────
X = df["email"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=500,
    stop_words="english",
    ngram_range=(1, 2),
    lowercase=True,
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print(f"\n Vocabulary size  : {len(vectorizer.vocabulary_)}")
print(f" Train samples    : {X_train_tfidf.shape[0]}")
print(f" Test  samples    : {X_test_tfidf.shape[0]}")

# ── 3. Train Models ──────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes"        : MultinomialNB(alpha=0.1),
}

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    results[name] = {
        "model" : model,
        "y_pred": y_pred,
        "acc"   : accuracy_score(y_test, y_pred),
    }

# -- 4. Results --------------------------------------------------------------
print("\n-- Model Accuracy --")
for name, r in results.items():
    print(f" {name:<25} : {r['acc']*100:.1f}%")

best_name  = "Logistic Regression"
best_model = results[best_name]["model"]
y_pred     = results[best_name]["y_pred"]

print(f"\n-- Classification Report ({best_name}) --")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# ── 5. Confusion Matrix Plot ─────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"])
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix — Spam Detector")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# ── 6. Top Spam Keywords ─────────────────────────────────────
feature_names = vectorizer.get_feature_names_out()
coef = best_model.coef_[0]
top_spam = pd.DataFrame({
    "word"  : feature_names,
    "weight": coef
}).nlargest(10, "weight")

print("\n-- Top 10 Spam Keywords --")
print(top_spam.to_string(index=False))

# -- 7. Predict New Emails ------------------------------------------------
print("\n-- Live Prediction --")
test_cases = [
    "You have won a free trip to Bali! Claim now!",
    "Please send me the project files by tomorrow morning.",
    "Discount medicine available. No prescription required.",
    "Hi, are you joining the team meeting at 4pm today?",
]

for email in test_cases:
    vec   = vectorizer.transform([email])
    pred  = best_model.predict(vec)[0]
    prob  = best_model.predict_proba(vec)[0][1]
    label = "SPAM" if pred == 1 else "HAM"
    print(f" [{label} {prob*100:.0f}%] {email[:55]}...")

print("\n Done!")
