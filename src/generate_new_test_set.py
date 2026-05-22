import json
import zipfile
from pathlib import Path
import sys

# Adjust the import path if your root directory structure requires it
from src.encipherment.cipher import HomophonicCipher


def main():
    keys_file = Path("extracted_test_keys.json")
    test_zip_path = Path("/ceph/project/SW10-CausalLM/Ciphers/Test/test_final.zip")
    output_file = Path("new_test_ciphers.jsonl")

    # 1. Load the pre-extracted training keys
    if not keys_file.exists():
        print(f"Error: Could not find {keys_file}. Run extract_keys.py first.")
        sys.exit(1)

    with open(keys_file, "r") as f:
        bucket_keys = json.load(f)

    needed_buckets = set(bucket_keys.keys())
    print(f"Loaded {len(needed_buckets)} target keys. Scanning test plaintexts...")

    # 2. Stream test plaintexts and apply keys
    with open(output_file, "w") as out_f:
        with zipfile.ZipFile(test_zip_path, "r") as z:
            for filename in z.namelist():
                if not filename.endswith(".jsonl"):
                    continue

                print(f"Scanning {filename}...")
                with z.open(filename, "r") as f:
                    for line in f:
                        if not needed_buckets:
                            break  # We found all 81 ciphers

                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Check if this plaintext belongs to a bucket we still need
                        length = data.get("length")
                        redundancy = data.get("redundancy")
                        bucket_id = f"{length}_{redundancy}"

                        if bucket_id in needed_buckets:
                            extracted_info = bucket_keys[bucket_id]
                            training_key = extracted_info["key"]
                            plaintext = data.get("plaintext", "")

                            # --- VERIFICATION STEP ---
                            # Ensure every character in the plaintext has a mapping in the training key.
                            # Since we filtered for len(key) >= 26 in the extraction script,
                            # this should always pass, but it prevents crashes if a weird character sneaks in.
                            missing_chars = [
                                char
                                for char in set(plaintext)
                                if char not in training_key
                            ]
                            if missing_chars:
                                continue  # Skip this plaintext and find another one for this bucket

                            # --- PREPARE CIPHER OBJECT ---
                            text_obj = {
                                "text": plaintext,
                                "text_with_boundaries": data.get(
                                    "plaintext_with_boundaries", plaintext
                                ),
                                "genres": data.get("genres", []),
                                "source_id": data.get("source_id", "unknown"),
                                "source_name": data.get("source_name", "unknown"),
                            }

                            # Initialize cipher normally (this assigns redundancy and text)
                            cipher = HomophonicCipher(text_obj, redundancy=redundancy)

                            # --- INJECT TRAINING KEY ---
                            cipher.key = training_key
                            cipher.num_symbols = extracted_info["target_mu"]

                            # --- ENCIPHER & SAVE ---
                            # This applies the key and handles your _apply_recurrence_and_remap_key() logic
                            cipher.encipher()

                            out_f.write(json.dumps(cipher.__json__()) + "\n")

                            needed_buckets.remove(bucket_id)
                            print(
                                f"  [+] Generated cipher for N={length}, Red={redundancy}. {len(needed_buckets)} left."
                            )

                if not needed_buckets:
                    break

    # 3. Final Check
    if needed_buckets:
        print(
            f"\nWarning: Could not find plaintexts for these buckets in the test zip: {needed_buckets}"
        )
    else:
        print(
            f"\nSuccess! Generated all 81 test ciphers. Saved to {output_file.resolve()}"
        )


if __name__ == "__main__":
    main()
