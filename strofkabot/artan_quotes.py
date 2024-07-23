""" artan.py
    This module defines Artan. A class that returns
    random Artan Kastro Quotes
"""
import pathlib
import yaml
import random

class ArtanQuotes():
    def __init__(self, quotes_file_path: pathlib.Path):
        self.quotes = self._extract_quotes(quotes_file_path)

    def _extract_quotes(self, quotes_file_path: pathlib.Path)-> list[str]:
        try:
            quotes = self._get_quotes_from_yaml_file(quotes_file_path)
        except FileNotFoundError as exc:
            raise ValueError(f"Unable to open quotes file: {quotes_file_path}") from exc

        if not isinstance(quotes, list):
            raise ValueError("A valid quote list hasn't been found")

        if len(quotes) == 0:
            raise ValueError("Quotes list is empty")

        return quotes

    def _get_quotes_from_yaml_file(self, quotes_file_path: pathlib.Path) -> list:
        with open(quotes_file_path, encoding='utf8') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise ValueError(f"Unable to parse yaml file: {quotes_file_path}") from exc

    def get_random_quote(self):
        """ Get a random quotes from the quotes list """ 
        index = random.randrange(len(self.quotes))
        return self.quotes[index]
