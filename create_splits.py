import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = r"F:\puja\1_desktop\PROJECTS\NeuroScan-AI\archive\BraTS2020_training_data\content\data\meta_data.csv"

# Load metadata
df = pd.read_csv(DATA_PATH)

# Get unique volumes
volumes = sorted(df["volume"].unique())

print("Total volumes:", len(volumes))

# 70% Train, 15% Validation, 15% Test
train_volumes, temp_volumes = train_test_split(
    volumes,
    test_size=0.30,
    random_state=42
)

val_volumes, test_volumes = train_test_split(
    temp_volumes,
    test_size=0.50,
    random_state=42
)

# Save CSV files
pd.DataFrame({"volume": train_volumes}).to_csv(
    "train_volumes.csv",
    index=False
)

pd.DataFrame({"volume": val_volumes}).to_csv(
    "val_volumes.csv",
    index=False
)

pd.DataFrame({"volume": test_volumes}).to_csv(
    "test_volumes.csv",
    index=False
)

print("Train volumes:", len(train_volumes))
print("Validation volumes:", len(val_volumes))
print("Test volumes:", len(test_volumes))

print("\nCSV files created successfully!")