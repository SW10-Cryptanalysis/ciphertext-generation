import os
import json
import zipfile
from pathlib import Path
import sys

# The exact 81 combinations we need to find in the training set
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


def main():
    training_dir = Path("/ceph/project/SW10-CausalLM/Ciphers/Training")
    output_file = Path("small_training_set.jsonl")

    # --- PHASE 1: INITIALIZE TARGETS ---
    targets = {}
    for length, redundancies in TEST_MATRIX.items():
        for red in redundancies:
            bucket_id = f"{length}_{red}"
            targets[bucket_id] = {
                "target_length": length,
                "target_redundancy": red,
                "found": False,
            }

    missing_ciphers = len(targets)
    print(f"Goal: Extract {missing_ciphers} ciphers from the training dataset.\n")
    print("Constraints:")
    print("  - Monoalphabetic (Red=0): Length ±25, Redundancy exact match.")
    print("  - Homophonic   (Red>0): Length ±10, Redundancy ±1.\n")

    if not training_dir.exists():
        print(f"Error: Could not find training directory at {training_dir}")
        sys.exit(1)

    # --- PHASE 2: SCAN AND EXTRACT ---
    with open(output_file, "w", encoding="utf-8") as out_f:
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
                                line_str = line.decode("utf-8")
                                train_data = json.loads(line_str)
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                continue

                            l = train_data.get("length")
                            r = train_data.get("redundancy")

                            if l is None or r is None:
                                continue

                            # Check if this cipher fits into any missing target bucket
                            matched_bucket = None
                            for bucket_id, info in targets.items():
                                if not info["found"]:
                                    target_len = info["target_length"]
                                    target_red = info["target_redundancy"]

                                    # Dynamic tolerance logic based on cipher type
                                    if target_red == 0:
                                        # Monoalphabetic constraints
                                        len_match = abs(l - target_len) <= 25
                                        red_match = r == 0
                                    else:
                                        # Homophonic constraints
                                        len_match = abs(l - target_len) <= 10
                                        red_match = abs(r - target_red) <= 1

                                    if len_match and red_match:
                                        matched_bucket = bucket_id
                                        break

                            # If it fits a needed bucket, save it
                            if matched_bucket:
                                out_f.write(line_str.strip() + "\n")
                                targets[matched_bucket]["found"] = True
                                missing_ciphers -= 1
                                print(
                                    f"  [+] Found N={l}, Red={r} (Fulfilled target {matched_bucket}). {missing_ciphers} left."
                                )

                    if missing_ciphers == 0:
                        break  # Break out of inner zip iteration

            if missing_ciphers == 0:
                break  # Break out of outer directory iteration

    # --- PHASE 3: WRAP UP ---
    if missing_ciphers > 0:
        print(
            f"\nWarning: {missing_ciphers} buckets couldn't be found in the training data."
        )
        for bucket, info in targets.items():
            if not info["found"]:
                print(f"  - Missing: Target {bucket}")
    else:
        print(
            f"\nSuccess! Extracted all 81 training ciphers. Saved to {output_file.resolve()}"
        )


if __name__ == "__main__":
    main()
