import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PROC_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def load_data():
    train = pd.read_parquet(PROC_DIR / "train.parquet")
    test = pd.read_parquet(PROC_DIR / "test.parquet")
    # 🔑 Fixed: "Message" matches your pipeline output
    return train["Message"], train["label"], test["Message"], test["label"]

def train_and_track():
    X_train, y_train, X_test, y_test = load_data()
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("sms_spam_detection")
    
    models = {
        "LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),
        "RandomForest": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42))
        ])
    }
    
    best_f1 = -1
    best_name = None
    
    for name, pipe in models.items():
        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            
            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds),
                "recall": recall_score(y_test, preds),
                "f1": f1_score(y_test, preds)
            }
            
            mlflow.log_params({"model_type": name, "max_features": 5000, "ngram_range": "(1,2)"})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, artifact_path="model")
            
            print(f"📊 {name} | F1: {metrics['f1']:.4f} | Acc: {metrics['accuracy']:.4f}")
            
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_name = name
                
    print(f"🏆 Best Model: {best_name} (F1: {best_f1:.4f})")
    
    best_run = mlflow.search_runs(order_by=["metrics.f1 DESC"], max_results=1)
    run_id = best_run.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/model"
    best_pipeline = mlflow.sklearn.load_model(model_uri)
    joblib.dump(best_pipeline, MODEL_DIR / "best_model.joblib")
    print("💾 Saved models/best_model.joblib")

if __name__ == "__main__":
    train_and_track()