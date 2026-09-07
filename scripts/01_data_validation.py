import mlflow
from sklearn.datasets import load_breast_cancer


def validate_data():
    """
    Loads the breast cancer dataset, performs validation checks
    including class balance, and logs the results to MLflow.
    """
    mlflow.set_experiment("Breast Cancer - Data Validation")

    with mlflow.start_run():
        print("Starting data validation run...")
        mlflow.set_tag("ml.step", "data_validation")

        # 1. Load data as a Pandas DataFrame
        cancer_data = load_breast_cancer(as_frame=True)
        df = cancer_data.frame
        print("Data loaded successfully.")

        # 2. Perform validation checks
        num_rows, num_cols = df.shape
        num_classes = df['target'].nunique()
        missing_values = df.isnull().sum().sum()

        # คำนวณ class balance (สัดส่วนของคลาสที่น้อยที่สุด)
        class_counts = df['target'].value_counts(normalize=True)
        min_class_ratio = class_counts.min()

        print(f"Dataset shape: {num_rows} rows, {num_cols} columns")
        print(f"Number of classes: {num_classes}")
        print(f"Missing values: {missing_values}")
        print(f"Minimum class ratio: {min_class_ratio:.4f}")

        # 3. Log validation results to MLflow
        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)
        mlflow.log_metric("class_balance", min_class_ratio)
        mlflow.log_param("num_classes", num_classes)

        # 4. ตรวจสอบเงื่อนไขทั้งหมด (Breast Cancer มี 2 คลาส ไม่ใช่ 3 เหมือน Wine)
        validation_status = "Success"
        if missing_values > 0:
            validation_status = "Failed"
        if num_classes != 2:
            validation_status = "Failed"
        if min_class_ratio < 0.20:
            validation_status = "Failed"

        mlflow.log_param("validation_status", validation_status)
        print(f"Validation status: {validation_status}")

        # 5. ทำให้ CI จับได้จริง — คืน exit code ที่ไม่ใช่ 0 เมื่อไม่ผ่าน
        if validation_status == "Failed":
            raise SystemExit("Data validation failed — หยุด pipeline ไม่ให้ไปขั้นถัดไป")

        print("Data validation run finished.")


if __name__ == "__main__":
    validate_data()