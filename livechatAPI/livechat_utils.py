import random
from collections import deque
import csv



def write_messages_csv(file_path, messages):
    '''
    intented for writing tuples of data (<user name>, <message>)
    can technically be any format
    '''
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(messages)


def read_messages_csv(file_path, num_messages=10):
    '''
    intented for reading tuples of data (<user name>, <message>)
    can technically be any format
    by default returns the latest 10 tuples of data as a list of tuples
    '''
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        messages = deque(csv_reader, maxlen=num_messages)
    return [tuple(row) for row in messages]


def append_message(chat_messages: list, user_msg: tuple):
    MAX_MESSAGES = 10
    if len(chat_messages) < MAX_MESSAGES:
        chat_messages.append(user_msg)
    else:
        chat_messages.pop(0)
        chat_messages.append(user_msg)

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
            print(lst)
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
                self.pick_counts[i] += 1  # Increment pick count for the chosen list
                return random.choice(self.lists[i])
