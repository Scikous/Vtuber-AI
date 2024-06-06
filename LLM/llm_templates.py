class BaseTemplate:
    def __init__(self, instructions_str="", user_name="user", character_name="assistant"):
        self.instructions_str = instructions_str
        self.user_name = user_name
        self.character_name = character_name

    def capybaraChatML(self, user_str="", context_str=""):
        raise NotImplementedError(
            "This method should be implemented by subclasses")


class PromptTemplate(BaseTemplate):
    def capybaraChatML(self, user_str="", context_str=""):
        return f'''<|im_start|>system
{self.instructions_str}<|im_end|>
<|im_start|>context
{context_str}
<|im_end|>
<|im_start|>{self.user_name}
{user_str}<|im_end|>
<|im_start|>{self.character_name}
'''


class DataTemplate(BaseTemplate):
    def capybaraChatML(self, user_str="", context_str="", character_str=""):
        return f'''<|im_start|>system
{self.instructions_str}<|im_end|>
<|im_start|>context
{context_str}<|im_end|>
<|im_start|>{self.user_name}
{user_str}<|im_end|>
<|im_start|>{self.character_name}
{character_str}<|im_end|>'''
