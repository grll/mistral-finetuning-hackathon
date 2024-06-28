import pickle


class Classifier:
    def __init__(self, model, tfidf_vectorizer, id2label):
        self.model = model
        self.tfidf_vectorizer = tfidf_vectorizer
        self.id2label = id2label

    def predict(self, text):
        X = self.tfidf_vectorizer.transform([text])
        y_pred = self.model.predict(X)
        return self.id2label[y_pred[0]]

    @classmethod
    def from_pretrained(cls, model_folder_path):
        with open(f"{model_folder_path}/model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(f"{model_folder_path}/tfidf_vectorizer.pkl", "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        with open(f"{model_folder_path}/id2label.pkl", "rb") as f:
            id2label = pickle.load(f)
        return cls(model, tfidf_vectorizer, id2label)
