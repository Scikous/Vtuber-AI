*** Settings ***
Documentation   Robot for custom ai voice generation
Library         SeleniumLibrary
Resource        utils.robot


*** Variables ***
#http://localhost:9874/
${url}          http://localhost:9874/
${browser}      edge
${input_folder_path}  wowzers
${ASR_model}  Faster Whisper (多语种)

*** Test Cases ***
Load Page
    Open browser    ${url}  ${browser}
    # ASR
    Set Gradio Textarea By Label    Input folder path   ${input_folder_path}
    Set Gradio Dropdown By Option    Faster Whisper (多语种)
    Set Gradio Dropdown By Option    en
    Set Gradio Dropdown By Option    large-v3
    Click Gradio Button         id    component-39
    Wait For Operation Completion    operation_type=ASR    timeout=14400s
    #UVR5
    Set Gradio Checkbox By Label    Open UVR5-WebUI
    # Switch to new address
    Go To    http://localhost:9873/
    Click Gradio Entity       checkbox  name    test
    Set Gradio Textarea By Label    Enter the path of the audio folder to be processed:   totalpath
    Set Gradio Dropdown By Option    HP5_only_main_vocal
    Click Gradio Entity     radio   name    radio-component-16
    Click Gradio Button         id    component-18

    #Formatting
    Go To    http://localhost:9874/
    Click Gradio Button         text    1-GPT-SOVITS-TTS
    Set Gradio Textarea By Label    *Experiment/model name   aName
    Set Gradio Textarea By Label    *Text labelling file   textlabelingfilepath
    Click Gradio Button         id    component-102
    Wait For Operation Completion    operation_type=One-click    timeout=14400s

    #training
    Click Gradio Button         text    1B-Fine-tuned training

    Click Gradio Button         id    component-118
    Wait For Operation Completion    operation_type=SoVITS     timeout=14400s
    
    Click Gradio Button         id    component-133
    Wait For Operation Completion    operation_type=GPT    timeout=14400s

    Sleep       15 Seconds
    [Teardown]  Close Browser