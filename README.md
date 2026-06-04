# 📧 Email Spam Detector

Classifies emails as **Spam or Ham** using TF-IDF vectorization and Logistic Regression, with a Naive Bayes comparison.

## Tech Stack
`Python` `Scikit-learn` `TF-IDF` `Logistic Regression` `Naive Bayes` `Matplotlib` `Seaborn`

## Features
- TF-IDF vectorization with bigrams
- Logistic Regression vs Naive Bayes comparison
- Confusion matrix visualization
- Top spam keyword extraction
- Live email prediction with confidence score

## Results

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | ~100%    |
| Naive Bayes         | ~90%     |

## How to Run

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
python spam_detector.py
```

## Sample Output
```
[SPAM 99%] You have won a free trip to Bali! Claim now!...
[HAM  2%]  Please send me the project files by tomorrow...
```

## Output Files
- `confusion_matrix.png` — prediction accuracy visualization

---
