import unittest
from message_filter import MessageFilter  # Assume the class is in a file named message_filter.py

class TestMessageFilter(unittest.TestCase):
    def setUp(self):
        self.filter = MessageFilter()

    def test_short_message(self):
        self.assertFalse(self.filter.is_valid_message("Short"))

    def test_valid_message(self):
        self.assertTrue(self.filter.is_valid_message("This is a valid message"))

    def test_message_with_http_link(self):
        self.assertFalse(self.filter.is_valid_message("Check out http://example.com"))
    
    def test_message_with_https_link(self):
        self.assertFalse(self.filter.is_valid_message(" https://www.reddit.com/r/science/comments/8cih30/a_new_study_suggests_that_romance_protects_gay/?utm_source=reddit-android, created at 2018-04-15T22:33:59.926000+00:00"))

    def test_message_with_emoji(self):
        self.assertFalse(self.filter.is_valid_message("<:GWqlabsBan:398950688555663360>"))
    
    def test_message_with_multiple_emojis(self):
        self.assertFalse(self.filter.is_valid_message("<:GWqlabsBan:398950688555663360> <:GWqlabsBan:398950688555663360>"))

    def test_message_with_tag(self):
        self.assertFalse(self.filter.is_valid_message("test <@!416623828920172544> tesdfsdfsdfst"))

    def test_message_with_colon(self):
        self.assertTrue(self.filter.is_valid_message("This message has a colon: but it's not an emoji"))

    def test_message_with_at(self):
        self.assertTrue(self.filter.is_valid_message("This message has an @ symbol but not a valid tag"))

    def test_edge_case_message_length(self):
        self.assertFalse(self.filter.is_valid_message("12345678911111"))  # 9 characters
        self.assertTrue(self.filter.is_valid_message("1234567890111111"))  # 10 characters

    def test_message_with_multiple_exclusions(self):
        self.assertFalse(self.filter.is_valid_message("@user1234 check http://example.com :smiley:"))

if __name__ == '__main__':
    unittest.main()