"""Data tests — ตรวจว่าข้อมูลยังหน้าตาเหมือนที่ตกลงไว้ ก่อนจะเอาไปเทรน"""
from sklearn.datasets import load_breast_cancer

df = load_breast_cancer(as_frame=True).frame


def test_schema():
    """คอลัมน์ต้องครบ 30 ฟีเจอร์ + target"""
    assert df.shape[1] == 31
    assert "target" in df.columns


def test_no_missing():
    assert df.isnull().sum().sum() == 0


def test_two_classes():
    assert df["target"].nunique() == 2


def test_class_balance():
    """คลาสน้อยสุดต้องไม่ต่ำกว่า 20%"""
    class_ratio = df["target"].value_counts(normalize=True).min()
    assert class_ratio >= 0.20