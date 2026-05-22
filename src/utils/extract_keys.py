import os
import json
import zipfile
from pathlib import Path

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
        return 26
    return round(length / redundancy)


def main():
    training_dir = Path("/ceph/project/SW10-CausalLM/Ciphers/Training")
    output_file = Path("extracted_test_keys.json")

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
    print(f"Looking for {missing_count} fully populated keys...")

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

                        key_dict = data.get("key", {})

                        # --- THE NEW STRICT FILTER ---
                        # 1. Must have all 26 letters mapped in the dictionary
                        # 2. NONE of those letters can have an empty list `[]`
                        if len(key_dict) < 26 or any(
                            len(homophones) == 0 for homophones in key_dict.values()
                        ):
                            continue

                        num_symbols = data.get("num_symbols")
                        if num_symbols is None:
                            num_symbols = sum(len(v) for v in key_dict.values())

                        for bucket_id, target_info in targets.items():
                            if (
                                target_info["key"] is None
                                and target_info["target_mu"] == num_symbols
                            ):
                                target_info["key"] = key_dict
                                missing_count -= 1
                                print(
                                    f"  [+] Found perfect key for N={target_info['length']}, Red={target_info['redundancy']} (mu={num_symbols}). {missing_count} left."
                                )
                                break

                if missing_count == 0:
                    break
        if missing_count == 0:
            break

    if missing_count > 0:
        print(f"Warning: Missing {missing_count} keys.")
    else:
        print("Success! Found all 81 perfect keys.")

    with open(output_file, "w") as f:
        json.dump(targets, f, indent=4)


if __name__ == "__main__":
    main()
