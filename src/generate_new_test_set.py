import os
import json
import zipfile
from pathlib import Path
import sys

from encipherment.cipher import HomophonicCipher, MonoalphabeticCipher

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
    test_zip_path = Path("/ceph/project/SW10-CausalLM/Ciphers/Test/test_final.zip")
    training_dir = Path("/ceph/project/SW10-CausalLM/Ciphers/Training")
    output_file = Path("new_test_ciphers.jsonl")

    # --- PHASE 1: INITIALIZE TARGETS ---
    targets = {}
    for length, redundancies in TEST_MATRIX.items():
        for red in redundancies:
            bucket_id = f"{length}_{red}"
            targets[bucket_id] = {
                "length": length,
                "redundancy": red,
                "target_mu": calculate_target_mu(length, red),
                "test_data": None,  # Will hold the JSON from the test zip
                "required_chars": None,  # Will hold the set() of characters used in the text
                "completed": False,  # Flag when we successfully generated it
            }

    missing_texts = len(targets)
    print(
        f"Phase 1: Extracting {missing_texts} test plaintexts from {test_zip_path.name}..."
    )

    # --- PHASE 2: COLLECT 81 TEST PLAINTEXTS ---
    if not test_zip_path.exists():
        print(f"Error: Could not find test zip at {test_zip_path}")
        sys.exit(1)

    with zipfile.ZipFile(test_zip_path, "r") as z:
        for filename in z.namelist():
            if not filename.endswith(".jsonl"):
                continue

            with z.open(filename, "r") as f:
                for line in f:
                    if missing_texts == 0:
                        break

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    l = data.get("length")
                    r = data.get("redundancy")
                    bucket_id = f"{l}_{r}"

                    # If this plaintext belongs to a bucket we need, grab it
                    if bucket_id in targets and targets[bucket_id]["test_data"] is None:
                        targets[bucket_id]["test_data"] = data
                        targets[bucket_id]["required_chars"] = set(
                            data.get("plaintext", "")
                        )
                        missing_texts -= 1

            if missing_texts == 0:
                break

    if missing_texts > 0:
        print(
            f"Warning: Could not find plaintexts for {missing_texts} buckets in the test set!"
        )
        sys.exit(1)

    print("All 81 plaintexts acquired. Moving to training data search...\n")

    # --- PHASE 3: FIND MATCHING KEYS AND GENERATE ---
    missing_ciphers = len(targets)
    print(f"Phase 3: Scanning training data for {missing_ciphers} matching keys...")

    with open(output_file, "w") as out_f:
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
                            if missing_ciphers == 0:
                                break

                            try:
                                train_data = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            key_dict = train_data.get("key", {})
                            num_symbols = train_data.get("num_symbols")
                            if num_symbols is None:
                                num_symbols = sum(len(v) for v in key_dict.values())

                            # Check this key against our needed buckets
                            for bucket_id, info in targets.items():
                                if info["completed"]:
                                    continue

                                # 1. Must match target_mu
                                if info["target_mu"] != num_symbols:
                                    continue

                                # 2. Must contain mappings for ALL characters present in the test plaintext
                                req_chars = info["required_chars"]
                                if not all(
                                    char in key_dict and len(key_dict[char]) > 0
                                    for char in req_chars
                                ):
                                    continue

                                # === MATCH FOUND! GENERATE THE CIPHER ===
                                test_data = info["test_data"]
                                plaintext = test_data.get("plaintext", "")
                                length = info["length"]
                                redundancy = info["redundancy"]

                                text_obj = {
                                    "text": plaintext,
                                    "text_with_boundaries": test_data.get(
                                        "plaintext_with_boundaries", plaintext
                                    ),
                                    "length": length,
                                    "target_length": length,
                                    "genres": test_data.get("genres", []),
                                    "source_id": test_data.get("source_id", "unknown"),
                                    "source_name": test_data.get(
                                        "source_name", "unknown"
                                    ),
                                }

                                # Initialize the correct cipher
                                if redundancy == 0:
                                    cipher = MonoalphabeticCipher(text_obj)
                                else:
                                    cipher = HomophonicCipher(
                                        text_obj, redundancy=redundancy
                                    )

                                # Inject the validated training key
                                cipher.key = key_dict
                                cipher.num_symbols = num_symbols

                                # Encipher and save
                                cipher.encipher()
                                out_f.write(json.dumps(cipher.__json__()) + "\n")

                                info["completed"] = True
                                missing_ciphers -= 1
                                print(
                                    f"  [+] Match found & generated! N={length}, Red={redundancy}. {missing_ciphers} left."
                                )
                                break  # Stop checking this training key so it's not reused

                    if missing_ciphers == 0:
                        break
            if missing_ciphers == 0:
                break

    # --- PHASE 4: WRAP UP ---
    if missing_ciphers > 0:
        print(
            f"\nWarning: Finished scanning, but {missing_ciphers} buckets couldn't find a matching key."
        )
        for b_id, info in targets.items():
            if not info["completed"]:
                print(
                    f"  - Missing key for: {b_id} (Needs {info['target_mu']} symbols)"
                )
    else:
        print(
            f"\nSuccess! Generated all 81 test ciphers. Saved to {output_file.resolve()}"
        )


if __name__ == "__main__":
    main()
