from models import VtuberExllamav2, LLMModelConfig
from huggingface_hub import snapshot_download
from model_utils import load_character, prompt_wrapper, contains_sentence_terminator
from time import perf_counter
import asyncio

# from general_utils import read_messages_csv

#get current character's information to use
character_info_json = "LLM_Wizard/characters/character.json"
instructions, user_name, character_name = load_character(character_info_json)

#set prompt template to follow current character's instruction set and name
instructions_string = f"""{instructions}"""

#LLM model to use
# main_model = "LLM_Wizard/qwen2.5-vl-finetune-merged2"#"turboderp/Qwen2.5-VL-7B-Instruct-exl2"#"unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit"#"TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"#"LLM_Wizard/CapybaraHermes-2.5-Mistral-7B-GPTQ"#"unsloth/Meta-Llama-3.1-8B"#"LLM/Llama-3-8B-Test" #'LLM/Meta-Llama-3.1-8B/'
main_model = "HuggingFaceTB/SmolLM2-135M-Instruct"#"turboderp/Qwen2.5-VL-7B-Instruct-exl2"#"unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit"#"TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"#"LLM_Wizard/CapybaraHermes-2.5-Mistral-7B-GPTQ"#"unsloth/Meta-Llama-3.1-8B"#"LLM/Llama-3-8B-Test" #'LLM/Meta-Llama-3.1-8B/'
tokenizer_model = "HuggingFaceTB/SmolLM2-135M-Instruct"#"Qwen/Qwen2.5-VL-7B-Instruct"#"unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit"#"TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"#"LLM_Wizard/CapybaraHermes-2.5-Mistral-7B-GPTQ"#"unsloth/Meta-Llama-3.1-8B"#"LLM/Llama-3-8B-Test" #'LLM/Meta-Llama-3.1-8B/'
revision ="main"#"8.0bpw"

#test using the exllamav2
async def exllamav2_test():
    model_config = LLMModelConfig (
        main_model=main_model,
        tokenizer_model=tokenizer_model,
        revision=revision,
        character_name=character_name,
        instructions=instructions,
        is_vision_model=False
    )
    # Character = VtuberExllamav2.load_model(config=model_config)#(generator, gen_settings, tokenizer, character_name)

    async with await VtuberExllamav2.load_model(config=model_config) as Character:
            
        images = [
        # {"file": "media/test_image_1.jpg"},
        # {"file": "media/test_image_2.jpg"},
        {"url": "https://media.istockphoto.com/id/1212540739/photo/mom-cat-with-kitten.jpg?s=612x612&w=0&k=20&c=RwoWm5-6iY0np7FuKWn8FTSieWxIoO917FF47LfcBKE="},
        # {"url": "https://i.dailymail.co.uk/1s/2023/07/10/21/73050285-12283411-Which_way_should_I_go_One_lady_from_the_US_shared_this_incredibl-a-4_1689019614007.jpg"},
        # {"url": "https://images.fineartamerica.com/images-medium-large-5/metal-household-objects-trevor-clifford-photography.jpg"}
    ]
        prompt = prompt_wrapper("Do you like coffee? also do you remember what i like? Make a very long response", "User is happy to talk with you")
        dummy_memory =[
      "As the crimson sun dipped below the jagged horizon, casting long, ethereal shadows across the ancient, crumbling ruins, a lone figure, cloaked in worn, travel-stained fabric, paused to contemplate the vast, silent expanse of the desolate wasteland stretching endlessly before them, a chilling premonition of trials yet to come slowly solidifying in the depths of their weary soul.",
      "The intricate symphony of urban life continued its relentless crescendo, with the incessant blare of car horns, the distant wail of sirens, and the muffled murmur of countless conversations weaving a complex tapestry of sound that underscored the profound isolation often experienced amidst the bustling anonymity of a sprawling metropolis.",
      "Scientists, meticulously analyzing the arcane data collected from the deepest recesses of the oceanic trenches, discovered astonishing, bioluminescent organisms exhibiting previously unknown adaptive mechanisms, providing tantalizing insights into the astonishing resilience of life in environments once deemed utterly inhospitable to any form of complex existence.",
      "Despite the overwhelming complexities and numerous unforeseen obstacles encountered during the arduous, multi-year development cycle, the dedicated team of engineers, fueled by an unyielding passion for innovation and an unwavering commitment to their ambitious vision, ultimately managed to revolutionize the nascent field of quantum computing with their groundbreaking, paradigm-shifting invention.",
    #   "The venerable oak tree, standing as an immutable sentinel through countless seasons, its gnarled branches reaching skyward like ancient, petrified arms, silently bore witness to the fleeting dramas of human endeavor unfolding beneath its rustling canopy, embodying a timeless wisdom far exceeding the ephemeral lifespan of any transient civilization."
    ] #["Ahahahahah", "wowozers", "i like coke", "great to hear!", "Ahahahahah", "wowozers", "i like coke", "great to hear!", "Ahahahahah", "wowozers", "i like coke", "great to hear!"]
        
        start = perf_counter()
        prompt = prompt_wrapper("Bob: Describe the image shown", "Alice: You need brain surgery")
        prompt = "Describe the image."
        #20ms+ for pure text, 60ms+ with maxed short-term-memory, 230-280ms+ with images (may be lower with smaller images, but also higher with different images)
        # response = await Character.dialogue_generator(prompt=prompt, conversation_history=dummy_memory, images=images, max_tokens=512)
        response = await Character.dialogue_generator(prompt="How", conversation_history=None, images=None, max_tokens=512)
        full_output = ""
        async for result in response:
            output = result.get("text", "")
            end = perf_counter()
            print(end-start)
            full_output += output
            # break
            # if contains_sentence_terminator(output):
            #     end = perf_counter()
            #     print(end-start, output)
            #     await Character.cancel_dialogue_generation()

            # print(output,  end = "")    
                # break

        # print(f"Prompts: {msg}\n\nRESPONSE:\n{response}\n\nTime Taken (Seconds): {end-start}")
        print(f"\n\nTime Taken (Seconds): {end-start}")
        print(full_output)

asyncio.run(exllamav2_test())
# #test using the standard huggingface loader
# def huggingface_test():
    
#     Character = VtuberLLM.load_model(character_name=character_name)#(generator, gen_settings, tokenizer, character_name)
#     # print(Character.tokenizer.eos_token, Character.tokenizer.bos_token)
#     # return
#     msg = """You MUST have the following in your output EXACTLY as written: "hello", 'wow't'"""
#     start = perf_counter()
#     response = asyncio.run(Character.dialogue_generator(prompt=msg))#, max_tokens=400))
#     end = perf_counter()
#     print(f"Prompts: {msg}\nResponses:\n{response}\n\nTime Taken (Seconds): {end-start}")
