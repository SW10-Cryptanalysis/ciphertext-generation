"""Constants for the cipher generation."""

from pathlib import Path
from fetching.dataset_extractor import DatasetConfig

MIN_DIFFICULTY = 4
MAX_DIFFICULTY = 300
ALPHABET = "abcdefghijklmnopqrstuvwxyz"

NUM_TRAINING_CIPHERS = 2_000_000
NUM_VALIDATION_CIPHERS = 10_000
NUM_TEST_CIPHERS = 10_000

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GENRE_MAP_PATH = PROJECT_ROOT / "data" / "book_genres.jsonl"

BOOK_IDS_VALIDATION = [
    "7241",  # Fables of La Fontaine
    "6593",  # History of Tom Jones, a Foundling
    "1342",  # Pride and Prejudice
    "1661",  # The Adventures of Sherlock Holmes
    "15399",  # The Interesting Narrative of the Life of Olaudah Equiano...
    "1998",  # Thus Spake Zarathustra
    "3207",  # Leviathan
    "120",  # Treasure Island
    "2160",  # The Expedition of Humphry Clinker
    "2542",  # A Doll's House
    "84",  # Frankenstein
    "7370",  # Second Treatise of Government
    "12",  # Through the Looking-Glass
    "2852",  # The Hound of the Baskervilles
    "76",  # The Adventures of Huckleberry Finn
    "2148",  # The Works of Edgar Allan Poe - Volume 2
    "1952",  # The Yellow Wallpaper
    "1259",  # Twenty Years After
    "394",  # Cranford
    "76939",  # The laws of contrast of color
    "829",  # Gulliver's Travels into Several Remote Nations of the World
    "60976",  # Rip Van Winkle
    "26184",  # Simple Sabotage Field Manual
    "2814",  # Dubliners
    "36034",  # White Nights and Other Stories
]

TOTAL_BOOKS = 55_454
DATASETS: list[DatasetConfig] = [
    {
        "path": "common-pile/project_gutenberg_filtered",
        "type": "project_gutenberg",
        "split_name": "train",
        "column": "text",
        "prefix": "pg",
        "fallback_genres": ["Other / Uncategorized"],
    },
    {
        "path": "common-pile/news_filtered",
        "type": "news",
        "split_name": "train",
        "column": "text",
        "prefix": "news",
        "fallback_genres": ["News"],
    },
    {
        "path": "common-pile/doab_filtered",
        "type": "doab",
        "split_name": "train",
        "column": "text",
        "prefix": "doab",
        "fallback_genres": ["Academic Books"],
    },
]

DEFAULT_TAXONOMY = {
    "Sci-Fi & Fantasy": ["science fiction", "fantasy", "science-fiction"],
    "Horror & Mystery": ["crime", "mystery", "detective", "thriller", "horror"],
    "History": ["history", "historical"],
    "Romance": ["romance"],
    "Philosophy & Religion": ["philosophy", "religion", "spirituality"],
    "Science & Tech": ["science", "technology", "mathematics"],
    "Classic & General Literature": [
        "classics of literature",
        "british literature",
        "american literature",
        "russian literature",
        "french literature",
        "literature - other",
        "bibliomania",
        "novels",
        "banned books",
        "bestsellers",
    ],
    "Mythology & Folklore": ["mythology", "legends", "folklore"],
    "Poetry & Drama": ["poetry", "plays", "dramas", "films", "opera"],
    "Short Stories & Essays": ["short stories", "essays", "letters", "speeches"],
    "Journalism & Periodicals": [
        "journals",
        "journalism",
        "magazine",
        "punchinello",
        "mirror of literature",
        "notes and queries",
    ],
    "Nature & Environment": ["environmental", "nature", "animals", "gardening"],
    "Humor": ["humour", "humor"],
    "Children & Young Adult": ["children", "young adult", "school stories"],
    "Adventure & Travel": ["adventure", "travel"],
    "Biographies & Memoirs": ["biographies", "memoirs"],
    "Education & Family": ["parenthood", "family", "teaching", "education"],
    "Hobbies & Lifestyle": [
        "how to",
        "cooking",
        "drinking",
        "cookbooks",
        "sports",
        "hobbies",
    ],
    "Social Sciences": [
        "archaeology",
        "anthropology",
        "language",
        "communication",
        "sociology",
        "gender",
        "sexuality",
        "erotica",
    ],
    "Fine Arts & Architecture": ["art", "architecture", "music", "brochure series"],
    "Psychology & Health": [
        "psychiatry",
        "psychology",
        "health",
        "medicine",
        "nutrition",
        "physiology",
    ],
    "Business & Economics": ["business", "management", "economics"],
    "Law & Politics": ["law", "criminology", "politics"],
    "Reference & Information": [
        "encyclopedias",
        "dictionaries",
        "reference",
        "reports",
        "proceedings",
    ],
}
