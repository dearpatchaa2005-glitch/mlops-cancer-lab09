import mlflow
from sklearn.datasets import load_breast_cancer


def load_and_predict():
    """
    Simulates a production scenario by loading a model using an alias
    from the MLflow Model Registry and using it for prediction on
    one sample from each class.
    """
    MODEL_NAME = "cancer-classifier-prod"
    MODEL_ALIAS = "staging"

    print(f"Loading model '{MODEL_NAME}' with alias '@{MODEL_ALIAS}'...")

    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(f"Please make sure a model version has the alias '@{MODEL_ALIAS}' in the MLflow UI.")
        return

    # โหลดข้อมูลพร้อมชื่อคลาส
    data = load_breast_cancer(as_frame=True)
    X = data.frame.drop('target', axis=1)
    y = data.frame['target']
    target_names = data.target_names  # ['malignant', 'benign']

    # หยิบตัวอย่างแรกของแต่ละคลาส (class 0 = malignant, class 1 = benign)
    idx_class0 = y[y == 0].index[0]
    idx_class1 = y[y == 1].index[0]
    sample_indices = [idx_class0, idx_class1]

    print("-" * 50)
    for idx in sample_indices:
        sample_data = X.loc[[idx]]
        actual_label = y.loc[idx]
        prediction = model.predict(sample_data)[0]

        actual_name = target_names[actual_label]
        predicted_name = target_names[prediction]
        correct = "Correct" if actual_label == prediction else "Incorrect"

        print(f"Sample index: {idx}")
        print(f"Actual: {actual_name} | Predicted: {predicted_name} | {correct}")
        print("-" * 50)


if __name__ == "__main__":
    load_and_predict()