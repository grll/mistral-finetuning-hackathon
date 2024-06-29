import pickle
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from prompts import PROMPT_CLASSIFIER



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

class LLMClassifier:
    def __init__(self, model_id: str, api_key: str):
        self.client = MistralClient(
            api_key=api_key,
        )
        self.model_id = model_id

    def predict(self, text: str):
        # send request to model
        answer = self.client.chat(messages=[ChatMessage(role="user", content=PROMPT_CLASSIFIER.replace("[CASE]", text))], model=self.model_id)
        pred = answer.choices[0].message.content
        # parse the prediction
        start_idx = pred.find("{")
        end_idx = pred.find("}")+1
        parsed = eval(pred[start_idx:end_idx])
        if isinstance(parsed["case_category"], str):
            return parsed["case_category"]
        elif isinstance(parsed["case_category"], (list, tuple)):
            return parsed["case_category"][0]
