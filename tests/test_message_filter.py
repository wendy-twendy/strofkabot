
""" Test file for message_filter module """

# pylint: disable=redefined-outer-name
import pytest
from strofkabot.message_filter import MessageFilter

@pytest.fixture
def message_filter():
    """
    Fixture to create an instance of MessageFilter.
    """
    return MessageFilter()

def test_short_message(message_filter):
    """
    Test that a short message is considered invalid.
    """
    assert not message_filter.is_valid_message("Short")

def test_valid_message(message_filter):
    """
    Test that a valid message is correctly identified.
    """
    assert message_filter.is_valid_message("This is a valid message")

def test_message_with_http_link(message_filter):
    """
    Test that a message containing an HTTP link is considered invalid.
    """
    assert not message_filter.is_valid_message("Check out http://example.com")

def test_message_with_https_link(message_filter):
    """
    Test that a message containing an HTTPS link is considered invalid.
    """
    assert not message_filter.is_valid_message(" https://www.reddit.com/r/science/comments/8cih30/a_new_study_suggests_that_romance_protects_gay/?utm_source=reddit-android, created at 2018-04-15T22:33:59.926000+00:00")

def test_message_with_emoji(message_filter):
    """
    Test that a message containing a single emoji is considered invalid.
    """
    assert not message_filter.is_valid_message("<:GWqlabsBan:398950688555663360>")

def test_message_with_multiple_emojis(message_filter):
    """
    Test that a message containing multiple emojis is considered invalid.
    """
    assert not message_filter.is_valid_message("<:GWqlabsBan:398950688555663360> <:GWqlabsBan:398950688555663360>")

def test_message_with_tag(message_filter):
    """
    Test that a message containing a user tag is considered invalid.
    """
    assert not message_filter.is_valid_message("test <@!416623828920172544> tesdfsdfsdfst")

def test_message_with_colon(message_filter):
    """
    Test that a message containing a colon, but not an emoji, is considered valid.
    """
    assert message_filter.is_valid_message("This message has a colon: but it's not an emoji")

def test_message_with_at(message_filter):
    """
    Test that a message containing an @ symbol, but not a valid tag, is considered valid.
    """
    assert message_filter.is_valid_message("This message has an @ symbol but not a valid tag")

def test_edge_case_message_length(message_filter):
    """
    Test the edge case for message length.
    """
    assert not message_filter.is_valid_message("12345678911111")  # 9 characters
    assert message_filter.is_valid_message("1234567890111111")  # 10 characters

def test_message_with_multiple_exclusions(message_filter):
    """
    Test that a message containing multiple exclusions (e.g., user tag, link, emoji) is considered invalid.
    """
    assert not message_filter.is_valid_message("@user1234 check http://example.com :smiley:")
