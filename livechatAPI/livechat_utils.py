import random
# import threading

# lock = threading.Lock() #may or may not be useful in avoiding messing with different livechats at wrong times

#each livechat should keep to a set limit avoiding favoritism
def append_livechat_message(chat_messages: list, user_msg: tuple|list):
    """
    adds the latest message to the running list of messages while maintaining a set max size (default is 10)
    
    for twitch and kick livechats
    """
    # global lock
    MAX_MESSAGES = 10
    # with lock:
        # print(chat_messages)
    def _livechat_appender(chat_messages, user_msg):
        if len(chat_messages) >= MAX_MESSAGES:
                chat_messages.pop(0)
        chat_messages.append(msg)
    if type(user_msg) is tuple:
        _livechat_appender(chat_messages, user_msg)
    elif type(user_msg) is list:
        for msg in user_msg:
            _livechat_appender(chat_messages, msg)
        

#takes lists of livechat messages and allows for selecting a random message between them -- YT: [], TW: [], Kick: [] 
class ChatPicker:
    def __init__(self, *lists):
        self.lists = lists
        self.pick_counts = [0] * len(lists)  # Initialize pick counts for each list

    #get the probabilities of picking between livechats -- technically works for any lists
    def calculate_probabilities(self):
        '''
        A livechat with more activity will generally have lower probability of being picked
        
        As one chat is picked multiple times in a row, other chats get higher probability of being picked 
        '''
        probabilities = {}
        total_prob = 0
        
        for i, lst in enumerate(self.lists):
            if len(lst) > 0:  # Only calculate probability for non-empty lists
                probability = 1 / (len(lst) + 1 + self.pick_counts[i])
                probabilities[i] = probability
                total_prob += probability
        
        # Normalize probabilities to ensure they sum up to 1.0
        if total_prob > 0:
            probabilities = {k: v / total_prob for k, v in probabilities.items()}
        else:
            probabilities = {k: 0 for k in range(len(self.lists))}

        return probabilities

    #picks a randomly between multiple livechats -- technically works for any lists
    def pick_rand_message(self):
        probabilities = self.calculate_probabilities()
        rand = random.random()
        cumulative_prob = 0
        for i, prob in probabilities.items():
            cumulative_prob += prob
            if rand < cumulative_prob:
                self.pick_counts[i] += 1  # Increment pick count for the chosen list -- lowers chances of being chosen again later
                random_chat_message = random.choice(self.lists[i])
                self.lists[i].remove(random_chat_message) #remove chosen message from list
                # print(self.lists)
                return random_chat_message


