*** Settings ***
Documentation   Robot for custom ai voice generation
Library         SeleniumLibrary
Resource        utils.robot


*** Variables ***
#http://localhost:9874/
${url}          http://localhost:9874/
${browser}      edge
${model_name}   goodname
*** Test Cases ***
Load Page
    Open browser    ${url}  ${browser}
    Click Gradio Button         text    1-GPT-SOVITS-TTS
    Capture Page Screenshot     training.jpg

    #training
    Click Gradio Button         text    1B-Fine-tuned training
    Set Gradio Textarea By Label    *Experiment/model name   ${model_name}

    Click Gradio Button         id    component-118
    Wait For Operation Completion    operation_type=SoVITS    timeout=14400s

    # Wait For Operation Completion    operation_type=SoVITS     timeout=14400s
    
    Click Gradio Button         id    component-133
    # Wait For Operation Completion    operation_type=GPT    timeout=14400s
    Capture Page Screenshot     training.jpg

    Sleep       15 Seconds
    [Teardown]  Close Browser