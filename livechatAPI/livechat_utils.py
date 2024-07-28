import random
from collections import deque
import csv
import threading

lock = threading.Lock() #may or may not be useful in avoiding messing with different livechats at wrong times

#save livechat message(s) for convenient data
def write_messages_csv(file_path, messages):
    '''
    intented for writing tuples of chat messages -> (<user name>, <message>)
    can technically be any format
    '''
    global lock

    with open(file_path, mode='a', newline='\n', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        with lock:
            if isinstance(messages, list):
                csv_writer.writerows(messages)
            else:#assumes single message -- for twitch and kick
                csv_writer.writerow(messages)    

#fetch last 10 messages from csv
def read_messages_csv(file_path, num_messages=10):
    '''
    intented for return tuples of chat messages -> (<user name>, <message>)
    can technically be any format
    by default returns the latest 10 chat messages as a list of tuples -> [(<user name>, <message>), ...]
    '''

    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        messages = deque(csv_reader, maxlen=num_messages)
    return [tuple(row) for row in messages]

#each livechat should keep to a set limit avoiding favoritism
def append_livechat_message(chat_messages: list, user_msg: tuple):
    """
    adds the latest message to the running list of messages while maintaining a set max size (default is 10)
    for twitch and kick livechats
    """

    global lock
    MAX_MESSAGES = 10
    with lock:
        # print(chat_messages)
        if len(chat_messages) >= MAX_MESSAGES:
            chat_messages.pop(0)
        chat_messages.append(user_msg)

#takes lists of chat messages and allows for selecting a random message between them 
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
