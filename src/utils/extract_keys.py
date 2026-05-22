import os
import json
import zipfile
from pathlib import Path

# The test matrix from your DatasetConfig
TEST_MATRIX = {
    350: [5, 10, 15, 0],
    400: [5, 10, 15, 0],
    450: [5, 10, 15, 20, 0],
    600: [5, 10, 15, 20, 0],
    800: [5, 10, 15, 20, 25, 30, 0],
    1000: [5, 10, 15, 20, 25, 30, 0],
    2000: [5, 10, 15, 20, 25, 30, 50, 0],
    4000: [5, 10, 15, 20, 25, 30, 50, 100, 0],
    6000: [5, 10, 15, 20, 25, 30, 50, 100, 150, 0],
    8000: [5, 10, 15, 20, 25, 30, 50, 100, 200, 300, 0],
    10000: [5, 10, 15, 20, 25, 30, 50, 100, 200, 300, 0],
}


def calculate_target_mu(length: int, redundancy: int) -> int:
    """Calculate the required number of homophones based on length and redundancy."""
    if redundancy == 0:
        return 26  # Monoalphabetic baseline
    return round(length / redundancy)


def main():
    training_dir = Path("/ceph/project/SW10-CausalLM/Ciphers/Training")
    output_file = Path("extracted_test_keys.json")

    # 1. Build the shopping list
    targets = {}
    for length, redundancies in TEST_MATRIX.items():
        for red in redundancies:
            target_mu = calculate_target_mu(length, red)
            bucket_id = f"{length}_{red}"
            targets[bucket_id] = {
                "length": length,
                "redundancy": red,
                "target_mu": target_mu,
                "key": None,
            }

    missing_count = len(targets)
    print(f"Starting extraction. Looking for {missing_count} distinct keys...")

    # 2. Stream the zip files
    for zip_name in os.listdir(training_dir):
        if not zip_name.endswith(".zip"):
            continue

        zip_path = training_dir / zip_name
        print(f"Scanning {zip_name}...")

        with zipfile.ZipFile(zip_path, "r") as z:
            for filename in z.namelist():
                if not filename.endswith(".jsonl"):
                    continue

                with z.open(filename, "r") as f:
                    for line in f:
                        if missing_count == 0:
                            break

                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Safely grab the key and count symbols
                        key_dict = data.get("key", {})

                        # Crucial safeguard: ensure the training key maps all 26 letters.
                        # If a key only maps 24 letters because the training plaintext was short,
                        # it will throw a KeyError when enciphering a test text containing the missing letters.
                        if len(key_dict) < 26:
                            continue

                        # Determine mu (num_symbols). Fallback to manual sum if not in metadata.
                        num_symbols = data.get("num_symbols")
                        if num_symbols is None:
                            num_symbols = sum(len(v) for v in key_dict.values())

                        # 3. Check off our shopping list
                        for bucket_id, target_info in targets.items():
                            if (
                                target_info["key"] is None
                                and target_info["target_mu"] == num_symbols
                            ):
                                target_info["key"] = key_dict
                                missing_count -= 1
                                print(
                                    f"  [+] Found key for N={target_info['length']}, Red={target_info['redundancy']} (mu={num_symbols}). {missing_count} left."
                                )
                                break  # Ensure this specific JSONL entry is only consumed by one bucket

                if missing_count == 0:
                    break
        if missing_count == 0:
            break

    # 4. Save the mapped targets
    if missing_count > 0:
        print(f"Warning: Finished scanning but still missing {missing_count} keys.")
    else:
        print("Success! Found all 81 keys.")

    with open(output_file, "w") as f:
        json.dump(targets, f, indent=4)

    print(f"Keys saved to {output_file.resolve()}")


if __name__ == "__main__":
    main()
