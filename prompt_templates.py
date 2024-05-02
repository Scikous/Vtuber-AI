class prompt_template():
    def __init__(self, instructions_str="", character_name="assistant") -> None:
        self.instructions_str = instructions_str
        self.character_name = character_name
        
    def capybaraChatML(self, user_str="", context_str=""):
        prompt_template=f'''
            <|im_start|>system
            {self.instructions_str}<|im_end|>
            <|im_start|>context
            {context_str}
            <|im_end|>
            <|im_start|>user
            {user_str}<|im_end|>
            <|im_start|>{self.character_name}
        '''
        return prompt_template