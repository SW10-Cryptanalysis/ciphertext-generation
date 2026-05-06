from typing_extensions import TypedDict
from typing import TYPE_CHECKING
from utils.formatting import format_text, clean_spaces

if TYPE_CHECKING:
    from datasets.iterable_dataset import IterableDataset


class Book(TypedDict):
    """A typed dictionary for book metadata."""

    id: str
    text: str
    source_name: str
    source_type: str
    fallback_genres: list[str]


class TextStream(TypedDict):
    """A text stream object.

    A text stream object contains the text, source ID, source name, and length.
    """

    text: str
    text_with_boundaries: str
    source_id: str
    source_name: str
    length: int
    target_length: int
    genres: list[str]


def extract_specific_chunk(
    text: str,
    target_len: int,
    start_idx: int,
    partition_size: int,
) -> tuple[str, str] | None:
    """Extract a chunk of text with exactly target_len alphabetic characters.

    Args:
        text (str): The raw usable text string.
        target_len (int): The exact number of a-z characters required.
        start_idx (int): The starting cursor for the partition.
        partition_size (int): The total character budget for this search zone.

    Returns:
        tuple[str, str] | None: (spaceless_text, underscored_text) or None if fails.

    """
    raw_window = text[start_idx : start_idx + partition_size]
    if not raw_window:
        return None

    formatted_window = format_text(raw_window)

    if len(clean_spaces(formatted_window)) < target_len:
        return None

    spaced_target_idx = find_spaceless_target_index(formatted_window, target_len)

    final_start, final_end = find_boundaries(formatted_window, 0, spaced_target_idx)

    final_chunk = formatted_window[final_start:final_end]
    unbounded = clean_spaces(final_chunk).strip()
    bounded = final_chunk.strip().replace(" ", "_")

    if not unbounded or not bounded:
        return None

    return unbounded, bounded


def find_boundaries(
    raw_text: str,
    start_raw_idx: int,
    target_length: int,
) -> tuple[int, int]:
    """Search for the boundaries of a target in the raw text.

    Args:
        raw_text (str): String to search in.
        start_raw_idx (int): Index to start searching from.
        target_length (int): Target length for the boundaries.

    Returns:
        tuple[int, int]: The start and end indices of the target within the raw text.

    """
    search_window = 200

    lookback_limit = max(0, start_raw_idx - search_window)
    first_space = raw_text.rfind(" ", lookback_limit, start_raw_idx)

    start_idx = first_space + 1 if first_space != -1 else start_raw_idx

    end_raw_idx = start_idx + target_length
    lookahead_limit = end_raw_idx + search_window

    end_idx = raw_text.find(" ", end_raw_idx, lookahead_limit)

    if end_idx == -1:
        end_idx = end_raw_idx

    return start_idx, end_idx


def randomize_stream(stream: "IterableDataset") -> "IterableDataset":
    """Randomize the order of the stream.

    Args:
        stream (IterableDataset): The stream to randomize.

    Returns:
        IterableDataset: The randomized stream.

    """
    return stream.shuffle(seed=42, buffer_size=100)


def find_spaceless_target_index(spaced_text: str, target_len: int) -> int:
    """Find the index in a spaced string that corresponds to a spaceless length.

    Args:
        spaced_text (str): String to search in.
        target_len (int): Target length for the index.

    Returns:
        int: The index in the spaced string that corresponds to the target length.

    """
    non_space_count = 0

    for i, char in enumerate(spaced_text):
        if char.strip():
            non_space_count += 1
        if non_space_count == target_len:
            return i

    return len(spaced_text)


def get_usable_text(raw_text: str, len_bounds: tuple[int, int]) -> str:
    """Get the usable text from the raw text."""
    total_len = len(raw_text)
    start_trim = min(int(total_len * 0.02), 10000)
    end_trim = min(int(total_len * 0.01), 5000)

    if total_len < len_bounds[1] * 3:
        usable_text = raw_text
    else:
        usable_text = raw_text[start_trim : total_len - end_trim]
    return usable_text


def get_source_genres(book: Book, genre_map: dict[str, list[str]]) -> list[str]:
    """Get the source genres for a book.

    Args:
        book (Book): The book to get the genres for.
        genre_map (dict[str, list[str]]): The genre map to use for filtering.

    Returns:
        list[str]: The source genres for the book.

    """
    if book.get("source_type") == "project_gutenberg":
        book_id = str(book.get("id", ""))
        id_val = book_id.removeprefix("pg:")
        return genre_map.get(id_val, book.get("fallback_genres", []))
    else:
        return book.get("fallback_genres", [])
