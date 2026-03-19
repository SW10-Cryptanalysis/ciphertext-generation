# Ciphertext Generation Pipeline

This project is a comprehensive pipeline for generating synthetic ciphertexts from literary sources. It is designed to produce large datasets for training and evaluating decipherment models. The pipeline handles everything from source text sampling and genre mapping to cipher generation and final data formatting.

## Key Features

- **Automated Genre Mapping**: Maps book data from the Gutendex API to a simplified genre taxonomy, allowing for genre-balanced dataset creation.
- **Configurable Cipher Generation**: Supports both homophonic and monoalphabetic substitution ciphers with fine-grained control over redundancy and length distributions.
- **Parallel Processing**: Utilizes a multi-process architecture to generate ciphers in parallel, significantly speeding up dataset creation.
- **Google Drive Integration**: Automatically uploads generated datasets and metadata to specified Google Drive folders.
- **Flexible Data Sampling**: Implements a sophisticated corpus sampler that extracts text chunks according to configurable length and distribution targets.
- **Robust Preprocessing**: Includes scripts to convert the raw JSON output into tokenized Arrow datasets, ready for use with modern ML frameworks.

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (for environment and dependency management)
- Google Cloud SDK (for Google Drive authentication)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SW10/ciphers.git
    cd ciphers/ciphertext-generation
    ```

2.  **Install dependencies:**
    Run the `uv sync` command to install the required Python packages into a virtual environment.
    ```bash
    uv sync
    ```

3.  **Authenticate with Google Drive:**
    Ensure you have authenticated with the gcloud CLI and have the necessary permissions to upload files to Google Drive. You will also need to set the following environment variables to point to the correct Google Drive folder IDs:
    - `FOLDER_ID_TRAIN`
    - `FOLDER_ID_VAL`
    - `FOLDER_ID_TEST`
    - `FOLDER_ID_METADATA`

## Usage

The pipeline is divided into several stages. Below are the instructions for running each part.

### 1. Genre Mapping

To build the genre map from the Gutendex API, run the `map_genres.py` script. This will fetch book metadata, map bookshelves to genres, and save the result to `data/book_genres.jsonl`.

```bash
uv run src/map_genres.py
```

### 2. Cipher Generation

To generate the full cipher dataset, run the `gen_training.py` script. This will orchestrate the entire generation process, from sampling texts to producing and uploading ciphers.

```bash
uv run src/gen_training.py
```

### 3. Preprocessing for Model Training

After generating the raw JSON data, you can convert it into a tokenized Arrow dataset using the `preprocess.py` script. This script tokenizes the plaintexts and ciphertexts and saves them in an efficient format for training.

To process the data with spaces preserved as special tokens:
```bash
uv run src/preprocess.py --spaces
```

To process the data without spaces:
```bash
uv run src/preprocess.py
```

## JSON Data Structure

Each generated cipher is saved as a JSON object with the following structure:

```json
{
    "plaintext": "thequickbrownfoxjumpsoverthelazydogpackmyboxwithfivedozenliquorjugs",
    "plaintext_with_boundaries": "the_quick_brown_fox_jumps_over_the_lazy_dog_pack_my_box_with_five_dozen_liquor_jugs",
    "length": 67,
    "num_symbols": 36,
    "redundancy": 2,
    "key": {
        "a": [
            28
        ],
        "b": [
            9
        ],
        "c": [
            7
        ],
        "d": [
            31
        ],
        "e": [
            26,
            3,
            36
        ],
        "f": [
            14
        ],
        "g": [
            32
        ],
        "h": [
            25,
            2
        ],
        "i": [
            6,
            34
        ],
        "j": [
            17
        ],
        "k": [
            8
        ],
        "l": [
            27
        ],
        "m": [
            19
        ],
        "n": [
            13
        ],
        "o": [
            22,
            33,
            15,
            11
        ],
        "p": [
            20
        ],
        "q": [
            4
        ],
        "r": [
            10,
            24
        ],
        "s": [
            21
        ],
        "t": [
            1,
            35
        ],
        "u": [
            18,
            5
        ],
        "v": [
            23
        ],
        "w": [
            12
        ],
        "x": [
            16
        ],
        "y": [
            30
        ],
        "z": [
            29
        ]
    },
    "ciphertext": "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 3 24 1 25 26 27 28 29 30 31 11 32 20 28 7 8 19 30 9 33 16 12 34 35 25 14 6 23 3 31 15 29 36 13 27 34 4 18 22 24 17 5 32 21",
    "ciphertext_with_boundaries": "1 2 3 _ 4 5 6 7 8 _ 9 10 11 12 13 _ 14 15 16 _ 17 18 19 20 21 _ 22 23 3 24 _ 1 25 26 _ 27 28 29 30 _ 31 11 32 _ 20 28 7 8 _ 19 30 _ 9 33 16 _ 12 34 35 25 _ 14 6 23 3 _ 31 15 29 36 13 _ 27 34 4 18 22 24 _ 17 5 32 21",
    "genres": [
        "Fiction"
    ],
    "source_id": "12345",
    "source_name": "Example Text"
}
```
## Configuration

The core behavior of the cipher generation is controlled by the `DatasetConfig` class in `src/cipher_generation/config.py`. This class allows you to define:
- **Number of ciphers** for training and validation splits.
- **Length distribution**: The percentage of ciphers to generate within `foundation`, `transition`, and `frontier` length ranges.
- **Test Matrix**: A precise mapping of text lengths to the redundancy levels that should be generated for the test set.
