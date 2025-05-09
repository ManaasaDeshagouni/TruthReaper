import json

def merge_datasets(file_list, output_file):
    combined_data = []

    for file_name in file_list:
        print(f"🔹 Loading {file_name}...")
        with open(file_name, "r") as f:
            data = json.load(f)
            combined_data.extend(data)

    print(f"\n✅ Total entries after merge: {len(combined_data)}")
    with open(output_file, "w") as f:
        json.dump(combined_data, f, indent=2)
    print(f"📂 Merged dataset saved as: {output_file}")

if __name__ == "__main__":
    # Example usage:
    files_to_merge = [
        "sequence_dataset.json",           # Real data
        "synthetic_full_lie.json",    
        "synthetic_full_truth.json"  
    ]

    # Remove files you don't have yet
    files_to_merge = [f for f in files_to_merge]

    merge_datasets(files_to_merge, "sequence_dataset_combined.json")