*** Settings ***
Documentation   Robot for custom ai voice generation
Library         SeleniumLibrary
Resource        utils.robot


*** Variables ***
#http://localhost:9874/
${url}          http://localhost:9874/
${browser}      edge
${model_name}   goodname
${text_labeling_file_path}  .list file_path
*** Test Cases ***
Load Page
    Open browser    ${url}  ${browser}
    Capture Page Screenshot     formatting.jpg
    #Formatting
    Click Gradio Button         text    1-GPT-SOVITS-TTS
    Set Gradio Textarea By Label    *Experiment/model name   ${model_name}
    Set Gradio Textarea By Label    *Text labelling file   ${text_labeling_file_path}
    Click Gradio Button         id    component-102
    Wait For Operation Completion    operation_type=One-click    timeout=15s
    Capture Page Screenshot     formatting2.jpg


    Sleep       15 Seconds
    [Teardown]  Close Browser