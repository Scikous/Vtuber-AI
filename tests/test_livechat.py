from unittest.mock import MagicMock
from livechatAPI.livechat import LiveChatController
from livechatAPI.livechat_utils import append_livechat_message, ChatPicker
from livechatAPI.youtube import YTLive
import pytest


#test that a random message is picked from multiple livechat sources
@pytest.mark.asyncio
async def test_live_chat_picker():
    test_livechat_messages = [("user1", "Hello!"), ("user11", "Hi!")]
    test_livechat2_messages = [("user2", "Haha!"), ("user22", "Womp!")]
    test_livechat3_messages = [("user3", "Harhar!"), ("user33", "Wow!")]
    test_chat_sources = [test_livechat_messages, test_livechat2_messages, test_livechat3_messages]
    total_num_msgs = test_chat_sources
    print(total_num_msgs)
    picker = ChatPicker(*test_chat_sources)
    # Flatten the test_chat_sources to create a combined list of all messages
    all_messages_before = [msg for chat in test_chat_sources for msg in chat]    
    result = picker.pick_rand_message()
    all_messages_after = [msg for chat in test_chat_sources for msg in chat]

    #check that the picked message actually existed and that it no longer exists in any of the chats
    assert result in all_messages_before and result not in all_messages_after 
    print(result)

#test that appending to a livechat source is done correctly
@pytest.mark.parametrize(
    "test_data, user_msgs, expected_length",
    [
        # Test case: msgs already in list == Max, incoming msgs > Max (10 vs 12) sum > 10
        ([("alr", "full")] * 10, [("wow", "woer"), ("wo2", "w2er"), ("w3w", "w3er"),
                                   ("w0ow", "woer"), ("wo2", "w2er"), ("w3w", "w3er"),
                                   ("w1ow", "woer"), ("wo2", "w2er"), ("w3w", "w3er"),
                                   ("w59w", "w59er"), ("w32", "w32er"), ("w32w", "w32er")], 10),

        # Test case: msgs already in list == incoming msgs (3 vs 3) sum < 10
        ([("alr", "full")] * 3, [("wow", "woer"), ("wo2", "w2er"), ("w3w", "w3er")], 6),

        # Test case: msgs already in list < incoming msgs (3 vs 9) sum >= 10
        ([("alr", "full")] * 3, [("wow", "woer"), ("wo2", "w2er"), ("w3w", "w3er"),
                                   ("wow", "woer"), ("wo2", "w2er"), ("w3w", "w3er"),
                                   ("wow", "woer"), ("wo2", "w2er"), ("w3w", "w3er")], 10),

        # Test case: msgs already in >= incoming msgs (10 vs 9) sum >= 10
        ([("alr", "full")] * 10, [("wow", "woer"), ("wo2", "w2er"), ("w3w", "w3er"),
                                   ("wow", "woer"), ("wo2", "w2er"), ("w3w", "w3er"),
                                   ("wow", "woer"), ("wo2", "w2er"), ("w3w", "w3er")], 10),
    ]
)
def test_append_livechat_message(test_data, user_msgs, expected_length):
    append_livechat_message(test_data, user_msgs)
    assert len(test_data) == expected_length
