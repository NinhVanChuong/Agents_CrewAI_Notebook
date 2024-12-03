# This file contains the core function to parse documents from multiple files.

import sys
from typing import Any
from pathlib import Path
from icecream import ic

sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader
from .utils import get_files_from_folder_or_file_paths, get_extractor

load_dotenv()

def parse_multiple_files(files_or_folder: list[str] | str) -> list[Document]:
    """
    Read the content of multiple files.

    Args:
        files_or_folder (list[str] | str): List of file paths or folder paths containing files.
    Returns:
        list[Document]: List of documents from all files.
    """
    if isinstance(files_or_folder, str):
        files_or_folder = [files_or_folder]

    valid_files = get_files_from_folder_or_file_paths(files_or_folder)

    if len(valid_files) == 0:
        raise ValueError("No valid files found.")

    ic(valid_files)

    file_extractor = get_extractor()

    documents = SimpleDirectoryReader(
        input_files=valid_files,
        file_extractor=file_extractor,
    ).load_data(show_progress=True)

    return documents