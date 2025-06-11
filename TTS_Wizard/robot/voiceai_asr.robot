*** Settings ***
Documentation   Robot for custom ai voice generation
Library         SeleniumLibrary
Resource        utils.robot


*** Variables ***
#http://localhost:9874/
${url}          http://localhost:9874/
${browser}      edge
${input_folder_path}  C:/Users/Santa_/Documents/GithubRepos/Vtuber-AI/voiceAI/dataset/johnsmith/johnsmith
${ASR_model}  Faster Whisper (多语种)

*** Test Cases ***
Load Page
    Open browser    ${url}  ${browser}
    Maximize Browser Window
    Capture Page Screenshot     asr
    # ASR
    Set Gradio Textarea By Label    Input folder path   ${input_folder_path}
    Set Gradio Dropdown By Option    Faster Whisper (多语种)
    Set Gradio Dropdown By Option    en
    Set Gradio Dropdown By Option    large-v3
    Click Gradio Button         id    component-39
    Wait For Operation Completion    operation_type=ASR    timeout=14400s
    Capture Page Screenshot         asr2.jpg
    #UVR5
    Set Gradio Checkbox By Label    Open UVR5-WebUI
    # Switch to new address
    Sleep       5 Seconds
    Go To    http://localhost:9873/
    Click Gradio Entity       checkbox  name    test
    Set Gradio Textarea By Label    Enter the path of the audio folder to be processed:   ${input_folder_path}
    Set Gradio Dropdown By Option    HP5_only_main_vocal
    Click Gradio Entity     radio   name    radio-component-16
    Click Gradio Button         id    component-18
    Capture Page Screenshot     asr3.jpg
    Sleep       15 Seconds
    [Teardown]  Close Browser