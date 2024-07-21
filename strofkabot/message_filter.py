import re 


class MessageFilter():
    def __init__(self):
        self.message_length = 15
        self.emoji_pattern = re.compile(r'<:.*?:[a-z,A-Z,0-9]*?>')
        self.tag_pattern = re.compile(r'<@.*?>')

    def is_valid_message(self, message: str)-> bool:
        if len(message) < self.message_length:
            return False

        return not self._excluded_content(message)

    def _excluded_content(self, content:str):
        return self.is_link(content) or self.is_emoji(content) or self.is_tag(content)

    def is_link(self, content:str):
        return "http" in content

    def is_emoji(self, content:str):
        return bool(self.emoji_pattern.search(content))

    def is_tag(self, content:str):
        return bool(self.tag_pattern.search(content))
    
    def is_channel(self, content:str):
        return bool(self.channel_pattern.search(content))
    
