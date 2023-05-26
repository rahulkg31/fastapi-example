import pickle

print("Loading model.......")
with open("model.pkl", "rb") as rf:
	clf = pickle.load(rf)

print("Loading vectorizer.......")
with open("vector.pkl", "rb") as rf:
	vectorizer = pickle.load(rf)

print("Predictions....")
test_data = ["to use your credit, click the wap link in the next txt message or click here", "hi, hope you are doing well"]
test_data_features = vectorizer.transform(test_data)
print(test_data)
print(clf.predict(test_data_features))