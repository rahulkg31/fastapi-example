import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

print("Loading data........")
data = pd.read_csv("spam_data.csv")

print("Extracting features..........")
data["sms"] = data["sms"].str.lower()
cv = CountVectorizer()
X = cv.fit_transform(data["sms"])

print("Model training.....")
X_train, X_test, y_train, y_test = train_test_split(X, data["classification"], test_size=0.30, random_state=42)
clf = MultinomialNB()
clf.fit(X_train,y_train)
print("Accuracy of Model", clf.score(X_test,y_test)*100, "%")


print("Saving model and vectorizer in pickel format.....")
with open("model.pkl", "wb") as wf:
    pickle.dump(clf, wf)

with open("vector.pkl", "wb") as wf:
    pickle.dump(cv, wf)

print("Done!")