import pandas as pd
import re
import random
from collections import defaultdict

# Define file paths
structured_data_file_path = 'C:/Users/naren/Industry Project/anomaly-detection/parser/HDFS_100k.log_structured.csv'
anomaly_label_path = "anomaly_label.csv"
hdfs_train_path = "hdfs_train"
hdfs_test_normal_path = "hdfs_test_normal"
hdfs_test_abnormal_path = "hdfs_test_abnormal"

# Load structured log data
df = pd.read_csv(structured_data_file_path)

# Regular expression to extract block IDs
block_pattern = re.compile(r"blk_[-]?\d+")

# Extract Block_ID from Content using regex and get the first match (if any)
df['Block_ID'] = df['Content'].apply(lambda x: block_pattern.findall(str(x))[0] if block_pattern.findall(str(x)) else None)

# Remove 'E' from EventId
df['EventId'] = df['EventId'].str.replace('E', '', regex=False)

# Create a dictionary mapping Block_ID to sequences of EventId
block_event_mapping = defaultdict(list)

for _, row in df.iterrows():
    block_id = row["Block_ID"]
    event_id = str(row["EventId"])
    if block_id and event_id:
        block_event_mapping[block_id].append(event_id)

# Load anomaly labels
label_df = pd.read_csv(anomaly_label_path)

# Convert label dataframe to a dictionary for quick lookup
label_dict = dict(zip(label_df["BlockId"], label_df["Label"]))

# Split data into lists
normal_instances, anomaly_instances = [], []

for block_id, event_sequence in block_event_mapping.items():
    line = f"{block_id},{' '.join(event_sequence)}\n"
    if block_id in label_dict:
        if label_dict[block_id] == "Normal":
            normal_instances.append(line)  # Store Normal instances
        else:
            anomaly_instances.append(line)  # Store Anomaly instances

# Train-test split for Normal instances (80% train, 20% test)
random.shuffle(normal_instances)  # Shuffle before splitting
split_idx = int(0.8 * len(normal_instances))  # 80% index

train_data = normal_instances[:split_idx]  # 80% Normal instances → Train
test_normal = normal_instances[split_idx:]  # 20% Normal instances → Test
test_abnormal = anomaly_instances  # All Anomaly instances → Test

# Save files
with open(hdfs_train_path, "w") as f:
    f.writelines(train_data)
with open(hdfs_test_normal_path, "w") as f:
    f.writelines(test_normal)
with open(hdfs_test_abnormal_path, "w") as f:
    f.writelines(test_abnormal)

print("Parsing complete. Train-test split done!")
