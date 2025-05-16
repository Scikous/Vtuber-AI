##IMPORTANT: LEGACY -- TO BE DELETED IN THE FUTURE
from textwrap import dedent
class BaseTemplate:
    def __init__(self, instructions_str="", user_name="user", character_name="assistant"):
        self.instructions_str = instructions_str
        self.user_name = user_name
        self.character_name = character_name

    def capybaraChatML(self, user_str="", context_str=""):
        """
        Generates Capybara ChatML format.

        Args:
            user_str (str): User's input
            context_str (str): chat log/general information about the situation
            character_str (str) -- DataTemplate exclusive: Character's response
             
        Returns:
            str: The formatted Capybara ChatML string.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses")

#a class which contains different types of LLM prompt templates
class PromptTemplate(BaseTemplate):
    def capybaraChatML(self, user_str="", context_str=""):
        return dedent(f'''
                    <|im_start|>system
                        {self.instructions_str}<|im_end|>
                    <|im_start|>context
                        {context_str}
                    <|im_end|>
                    <|im_start|>{self.user_name}
                        {user_str}<|im_end|>
                    <|im_start|>{self.character_name}
''')

#a class which contains different types of LLM fine-tuning data templates
class DataTemplate(BaseTemplate):

    def capybaraChatML(self, user_str="", context_str="", character_str=""):
        return f'''
<s><|im_start|>system
{self.instructions_str}<|im_end|>
<|im_start|>context
{context_str}<|im_end|>
<|im_start|>{self.user_name}
{user_str}<|im_end|>
<|im_start|>{self.character_name}
{character_str}<|im_end|>
        '''