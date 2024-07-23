""" Test file for artan_quotes module """
import os
import pathlib
from typing import Union
import yaml
import pytest
from strofkabot.artan_quotes import ArtanQuotes
class TestArtanQuotes:
    """ Test class for ArtanQuotes class"""

    QUOTES = ["Quote 0","Quote 1"]

    def _create_quotes_file(self, tmp_path: pathlib.Path, quotes: Union[list,str]) -> pathlib.Path:
        artan_quote_file_path = tmp_path / "tmp_quotes_file"

        with open(artan_quote_file_path, 'w', encoding="utf8") as quote_file:
            yaml.dump(quotes, quote_file)

        return artan_quote_file_path

    def _get_path_quote_file_not_exists(self, tmp_path: pathlib.Path)->pathlib.Path:
        return tmp_path / "quote_files_does_not_exist.txt"

    def _get_path_quotes_file_empty(self, tmp_path: pathlib.Path)->pathlib.Path:
        empty_file_path = tmp_path / "quote_files_missing.txt"
        empty_file_path.touch()
        return empty_file_path

    def _get_path_quotes_file_invalid(self, tmp_path: pathlib.Path)->pathlib.Path:
        return self._create_quotes_file(tmp_path, "blalba")

    def _get_path_quotes_file_list_empty(self, tmp_path: pathlib.Path)->pathlib.Path:
        return self._create_quotes_file(tmp_path, [])

    def _get_single_quote_as_list(self)->list:
        quote_index = 0
        return [self.QUOTES[quote_index]]

    def _get_multiple_quotes_as_list(self)->list:
        return self.QUOTES

    def _create_single_quote_file(self, tmp_path: pathlib.Path)-> pathlib.Path:
        return self._create_quotes_file(tmp_path, self._get_single_quote_as_list())

    def _create_multiple_quotes_file(self, tmp_path: pathlib.Path)-> pathlib.Path:
        return self._create_quotes_file(tmp_path, self._get_multiple_quotes_as_list())

    def test_quotes_file_not_exists(self, tmp_path):
        """ Test missing quotes file """
        with pytest.raises(ValueError):
            # pylint: disable=unused-variable
            artan_missing_quotes = ArtanQuotes(
                quotes_file_path = self._get_path_quote_file_not_exists(tmp_path))

    def test_quotes_file_is_empty(self, tmp_path):
        """ Test case for an empty file""" 
        with pytest.raises(ValueError):
            # pylint: disable=unused-variable
            artan_empty_quotes_file = ArtanQuotes(
                quotes_file_path = self._get_path_quotes_file_empty(tmp_path))

    def test_invalid_quotes_file(self, tmp_path):
        """ Test quotes file not containts a list """ 
        with pytest.raises(ValueError):
            # pylint: disable=unused-variable
            artan_invalid_quotes_file = ArtanQuotes(
                quotes_file_path = self._get_path_quotes_file_invalid(tmp_path))

    def test_no_quotes_in_list(self, tmp_path):
        """ Test case for a scenario where quotes list is empty""" 
        with pytest.raises(ValueError):
            # pylint: disable=unused-variable
            artan_invalid_quotes_file = ArtanQuotes(
                quotes_file_path = self._get_path_quotes_file_list_empty(tmp_path))

    def test_get_quote_from_single_quote_file(self, tmp_path: pathlib.Path) -> None:
        """ Tests get random quote with a single quote file""" 
        single_quote_file_path = self._create_single_quote_file(tmp_path)
        artan_quote_generator = ArtanQuotes(quotes_file_path = single_quote_file_path)
        assert artan_quote_generator.get_random_quote() == self._get_single_quote_as_list()[0]

    def test_get_quote_from_multiple_quotes_file(self, tmp_path: pathlib.Path) -> None:
        """ Tests get random quote with from multiple quotes file"""
        multiple_quotes_file_path = self._create_multiple_quotes_file(tmp_path)
        artan_quote_generator = ArtanQuotes(quotes_file_path = multiple_quotes_file_path)
        assert artan_quote_generator.get_random_quote() in self._get_multiple_quotes_as_list()
