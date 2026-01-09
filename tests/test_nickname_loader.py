"""Tests for the nickname loader utility."""

from pathlib import Path

from strofkabot.utils.nickname_loader import get_display_name, load_nicknames


class TestLoadNicknames:
    """Tests for load_nicknames function."""

    def test_load_valid_nicknames_file(self, tmp_path: Path):
        """Should load nicknames from a valid YAML file."""
        nicknames_file = tmp_path / "nicknames.yaml"
        nicknames_file.write_text(
            """nicknames:
  123456789:
    - nick1
    - nick2
  987654321:
    - single_nick
"""
        )

        result = load_nicknames(nicknames_file)

        assert result == {
            123456789: ["nick1", "nick2"],
            987654321: ["single_nick"],
        }

    def test_load_empty_nicknames_file(self, tmp_path: Path):
        """Should return empty dict for file with empty nicknames."""
        nicknames_file = tmp_path / "nicknames.yaml"
        nicknames_file.write_text("nicknames:\n")

        result = load_nicknames(nicknames_file)

        assert result == {}

    def test_load_missing_file(self, tmp_path: Path):
        """Should return empty dict when file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.yaml"

        result = load_nicknames(nonexistent)

        assert result == {}

    def test_load_malformed_yaml(self, tmp_path: Path):
        """Should return empty dict for malformed YAML."""
        nicknames_file = tmp_path / "nicknames.yaml"
        nicknames_file.write_text("this is not: valid: yaml: [")

        result = load_nicknames(nicknames_file)

        assert result == {}

    def test_load_missing_nicknames_key(self, tmp_path: Path):
        """Should return empty dict when 'nicknames' key is missing."""
        nicknames_file = tmp_path / "nicknames.yaml"
        nicknames_file.write_text("other_key:\n  - value\n")

        result = load_nicknames(nicknames_file)

        assert result == {}


class TestGetDisplayName:
    """Tests for get_display_name function."""

    def test_user_has_nickname(self):
        """Should return first nickname when user is in the dict."""
        nicknames = {
            123456789: ["papa", "dad"],
            987654321: ["shark", "sharku"],
        }

        result = get_display_name(123456789, "Original Name", nicknames)

        assert result == "papa"

    def test_user_not_in_nicknames(self):
        """Should return fallback when user is not in the dict."""
        nicknames = {
            123456789: ["papa"],
        }

        result = get_display_name(999999999, "Fallback Name", nicknames)

        assert result == "Fallback Name"

    def test_empty_nicknames_dict(self):
        """Should return fallback when nicknames dict is empty."""
        result = get_display_name(123456789, "Fallback Name", {})

        assert result == "Fallback Name"

    def test_none_nicknames(self):
        """Should return fallback when nicknames is None."""
        result = get_display_name(123456789, "Fallback Name", None)

        assert result == "Fallback Name"

    def test_empty_nickname_list(self):
        """Should return fallback when user's nickname list is empty."""
        nicknames = {
            123456789: [],
        }

        result = get_display_name(123456789, "Fallback Name", nicknames)

        assert result == "Fallback Name"
