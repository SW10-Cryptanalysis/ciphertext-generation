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
            targets[bucket_id] = False  # False indicates we haven't found a match yet

    missing_ciphers = len(targets)
    print(f"Goal: Extract {missing_ciphers} ciphers from the training dataset.\n")

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
                                # Decode bytes to string
                                line_str = line.decode("utf-8")
                                train_data = json.loads(line_str)
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                continue

                            # Extract criteria
                            l = train_data.get("length")
                            r = train_data.get("redundancy")
                            bucket_id = f"{l}_{r}"

                            # If it's a target we need and haven't collected yet
                            if bucket_id in targets and not targets[bucket_id]:
                                # Write the unmodified JSON directly to our new file
                                out_f.write(line_str.strip() + "\n")

                                # Mark as found
                                targets[bucket_id] = True
                                missing_ciphers -= 1
                                print(
                                    f"  [+] Found N={l}, Red={r}. {missing_ciphers} left."
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
        for bucket, found in targets.items():
            if not found:
                print(f"  - Missing: Length_Redundancy = {bucket}")
    else:
        print(
            f"\nSuccess! Extracted all 81 training ciphers. Saved to {output_file.resolve()}"
        )


if __name__ == "__main__":
    main()
