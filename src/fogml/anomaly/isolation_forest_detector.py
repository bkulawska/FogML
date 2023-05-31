from sklearn.ensemble import IsolationForest
from sklearn.tree import export_text


class IsolationForestAnomalyDetector:
    def __init__(self, n_estimators=1, max_samples='auto', random_state=42):
        self.clf = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                   random_state=random_state)

    def fit(self, x):
        self.clf.fit(x)
        tree_text = export_text(self.clf.estimators_[0])
        print(tree_text)  

    def predict(self, x):
        return self.clf.predict(x)
