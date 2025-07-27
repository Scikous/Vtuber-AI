from general_utils import write_messages_csv, read_messages_csv
import asyncio
import pytest
import os
import gc
@pytest.fixture
def conversation_log_file_path():
    conversation_log_file = 'livechatAPI/data/test_conv_log.csv'        
    #check if path is full path (absolute path or not) -- add root directory extension if not abs path
    if not os.path.isabs(conversation_log_file):
        project_root = os.path.dirname(os.path.dirname(__file__)) + '/'
        conversation_log_file = project_root + conversation_log_file
    yield conversation_log_file

    del conversation_log_file
    gc.collect()

#runs first to create test file with test data -- data can be used for custom finetuning
@pytest.mark.run(order=1)
@pytest.mark.asyncio
async def test_write_messages_csv(conversation_log_file_path):
    test_user_llm_data = [("user1: Do you edge!", "I am the master of edging!"), ("user1: Skibidi rizz!", "erm, what the sigma!?")]
    conversation_log_file = conversation_log_file_path       
    for msg in test_user_llm_data:
        await write_messages_csv(conversation_log_file, message_data=msg)
    assert os.path.exists(conversation_log_file)
    assert os.path.getsize(conversation_log_file) > 0

#tries to read custom finetuning data from file and afterwards deletes file
@pytest.mark.run(order=2)
@pytest.mark.asyncio
async def test_read_messages_csv(conversation_log_file_path):
    conversation_log_file = conversation_log_file_path     
    #check if path is full path (absolute path or not) -- add root directory extension if not abs path
    if not os.path.isabs(conversation_log_file):
        project_root = os.path.dirname(os.path.dirname(__file__)) + '/'
        conversation_log_file = project_root + conversation_log_file
    msg = await read_messages_csv(conversation_log_file, num_messages=1)
    # print(msg)
    
    assert os.path.exists(conversation_log_file)

    #expected to be a tuple -- ('user1: Skibidi rizz!', 'erm, what the sigma!?')
    assert isinstance(msg[0], tuple)
    assert msg[0][0] == 'user1: Skibidi rizz!' and msg[0][1] == 'erm, what the sigma!?'
    os.remove(conversation_log_file)
    assert not os.path.exists(conversation_log_file)