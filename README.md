# ciphertext-generation
Generation of ciphertexts for us in deciphering - both for training and evaluation.

## Get started
1. Run the "uv sync" command to install/update dependencies

## Run the code
To generate data and upload to Google Drive, run `uv run src/gen_training.py`

## JSON Structure

```json
{
    "plaintext": "something",
    "plaintext_with_boundaries": "some_thing",
    "length": 9,
    "num_symbols": 9,
    "difficulty": 1,
    "key": {
        "a": [],
        "b": [],
        "c": [],
        "d": [],
        "e": [
            4,
        ],
        "f": [],
        "g": [
            9,
        ],
        "h": [
            6,
        ],
        "i": [
            7,
        ],
        "j": [],
        "k": [],
        "l": [],
        "m": [
            3,
        ],
        "n": [
            8,
        ],
        "o": [
            2,
        ],
        "p": [],
        "q": [],
        "r": [],
        "s": [
            1
        ],
        "t": [
            5,
        ],
        "u": [],
        "v": [],
        "w": [],
        "x": [],
        "y": [],
        "z": [],
    },
    "ciphertext": "1 2 3 4 5 6 7 8 9",
    "ciphertext_with_boundaries": "1 2 3 4 _ 5 6 7 8 9"
    "source_id": "35450",
    "source_name": "Our Cats and All About Them\r\nTheir Varieties, Habits, and Management; and for Show, the Standard of Excellence and Beauty; Described and Pictured",
    "genres": ["Sci-Fi & Fantasy"]
}

```
