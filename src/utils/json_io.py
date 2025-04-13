import json
import os


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def load_all_stocks(path):
    """
    Generator function to load all stock data from JSON files in a given directory.

    Parameters
    ----------
    path : str, optional
        Path to the directory containing stock JSON files (default is 'stocks').

    Yields
    ------
    dict
        Parsed JSON content of each file as a Python dictionary.

    Notes
    -----
    - Each file in the directory is expected to be a valid JSON file.
    - Files are read one at a time, making this function memory-efficient for large datasets.
    """

    for fname in os.listdir(path):
        file_path = path / fname
        yield read_json(file_path)
