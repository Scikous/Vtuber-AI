*** Keywords ***

Set Gradio Textarea By Label
    [Arguments]    ${label_text}    ${new_value}
    
    # Perform the action and log details
    Execute JavaScript    
    ...    (function() {
    ...        var shadowRoot = document.querySelector('gradio-app').shadowRoot;
    ...        var textareas = Array.from(shadowRoot.querySelectorAll('textarea[data-testid="textbox"]'));
    ...        var foundMatch = false;
    ...        
    ...        for (var i = 0; i < textareas.length; i++) {
    ...            var textarea = textareas[i];
    ...            var parentLabel = textarea.closest('label');
    ...            var labelSpan = parentLabel ? parentLabel.querySelector('span') : null;
    ...            
    ...            if (labelSpan) {
    ...                console.log('Current label:', labelSpan.textContent.trim());
    ...                if (labelSpan.textContent.trim() === '${label_text}') {
    ...                    console.log('Match found, setting value');
    ...                    textarea.focus();
    ...                    textarea.value = '${new_value}';
    ...                    textarea.dispatchEvent(new Event('input', {bubbles: true}));
    ...                    textarea.dispatchEvent(new Event('change', {bubbles: true}));
    ...                    foundMatch = true;
    ...                    break;
    ...                }
    ...            }
    ...        }
    ...        
    ...        if (!foundMatch) {
    ...            console.log('No matching textarea found for label: ${label_text}');
    ...        }
    ...    })()
    
    # Optional: Verify the action or add additional logging
    Log    Attempted to set textarea with label '${label_text}' to '${new_value}'

Set Gradio Checkbox By Label
    [Arguments]    ${label_text}    ${desired_state}=true
    
    Execute JavaScript    
    ...    (function() {
    ...        var shadowRoot = document.querySelector('gradio-app').shadowRoot;
    ...        var checkboxes = Array.from(shadowRoot.querySelectorAll('input[type="checkbox"]'));
    ...        
    ...        for (var i = 0; i < checkboxes.length; i++) {
    ...            var checkbox = checkboxes[i];
    ...            var parentLabel = checkbox.closest('label');
    ...            var labelSpan = parentLabel ? parentLabel.querySelector('span') : null;
    ...            
    ...            if (labelSpan) {
    ...                if (labelSpan.textContent.trim() === '${label_text}') {
    ...                    var isChecked = checkbox.checked;
    ...                    var shouldBeChecked = ${desired_state} === true;
    ...                    
    ...                    if (isChecked !== shouldBeChecked) {
    ...                        checkbox.click();
    ...                        checkbox.dispatchEvent(new Event('change', {bubbles: true}));
    ...                    }
    ...                    break;
    ...                }
    ...            }
    ...        }
    ...    })()
    
    # Optional: Add a small wait to ensure the change takes effect
    Sleep    1s

Set Gradio Dropdown By Option
    [Arguments]    ${option_text}
    
    Execute JavaScript    
    ...    (function() {
    ...        var shadowRoot = document.querySelector('gradio-app').shadowRoot;
    ...        
    ...        var selects = shadowRoot.querySelectorAll('select');
    ...        
    ...        for (var i = 0; i < selects.length; i++) {
    ...            var select = selects[i];
    ...            var parentLabel = select.closest('label');
    ...            var labelSpan = parentLabel ? parentLabel.querySelector('span') : null;
    ...            
    ...            for (var j = 0; j < select.options.length; j++) {
    ...                if (select.options[j].text.trim() === '${option_text}') {
    ...                    select.selectedIndex = j;
    ...                    select.dispatchEvent(new Event('change', {bubbles: true}));
    ...                    break;
    ...                }
    ...            }
    ...        }
    ...    })()
    
    # Optional: Add a small wait to ensure the change takes effect
    Sleep    1s


Click Gradio Button
    [Arguments]    ${search_type}    ${search_value}
    
    Execute JavaScript    
    ...    (function() {
    ...        var shadowRoot = document.querySelector('gradio-app').shadowRoot;
    ...        var button;
    ...        
    ...        switch ('${search_type}') {
    ...            case 'id':
    ...                button = shadowRoot.querySelector('button[id="${search_value}"]');
    ...                break;
    ...            case 'text':
    ...                var buttons = Array.from(shadowRoot.querySelectorAll('button'));
    ...                button = buttons.find(btn => btn.textContent.trim() === '${search_value}');
    ...                break;
    ...            case 'data-testid':
    ...                button = shadowRoot.querySelector('button[data-testid="${search_value}"]');
    ...                break;
    ...            case 'aria-label':
    ...                button = shadowRoot.querySelector('button[aria-label="${search_value}"]');
    ...                break;
    ...            default:
    ...                console.log('Unsupported search type: ${search_type}');
    ...                return;
    ...        }
    ...        
    ...        if (button) {
    ...            button.click();
    ...            console.log('Button clicked successfully');
    ...        } else {
    ...            console.log('Button not found with ${search_type}: ${search_value}');
    ...        }
    ...    })()


 

Click Gradio Entity
    [Arguments]     ${search_type}    ${search_element}    ${search_value}
    
    Execute JavaScript    
    ...    (function() {
    ...        var shadowRoot = document.querySelector('gradio-app').shadowRoot;
    ...        var entity;
    ...        
    ...        switch ('${search_element}') {
    ...            case 'name':
    ...                entity = shadowRoot.querySelector('input[type="${search_type}"][name="${search_value}"]');
    ...                break;
    ...            default:
    ...                console.log('Unsupported search type: ${search_type}');
    ...                return;
    ...        }
    ...        
    ...        if (entity) {
    ...            entity.checked = true;
    ...            console.log('Entity clicked successfully');
    ...        } else {
    ...            console.log('Entity not found with ${search_type}: ${search_value}');
    ...        }
    ...    })()


Check Log
    [Arguments]    ${operation_type}     ${label_extension}     ${operation_log_text}

    ${log_text} =    Execute JavaScript
    ...    return new Promise(resolve => {
    ...        const checkTextarea = () => {
    ...            var shadowRoot = document.querySelector('gradio-app').shadowRoot;
    ...            var textareas = Array.from(shadowRoot.querySelectorAll('textarea[data-testid="textbox"]'));
    ...            for (var i = 0; i < textareas.length; i++) {
    ...                var textarea = textareas[i];
    ...                var parentLabel = textarea.closest('label');
    ...                var labelSpan = parentLabel ? parentLabel.querySelector('span') : null;
    ...                if (labelSpan && labelSpan.textContent.trim() === '${label_extension}') {
    ...                    if (textarea.value !== '') {
    ...                        resolve(textarea.value);
    ...                        return;
    ...                    }
    ...                    else{
    ...                        console.log("Textarea is empty, waiting");
    ...                    }
    ...                }
    ...            }
    ...            setTimeout(checkTextarea, 100);
    ...        };
    ...        checkTextarea();
        # ...        
        # ...        return '';
    ...    });
    
    Should Contain    ${log_text}    ${operation_log_text}    msg=Training has not completed
    
Wait For Operation Completion
    [Arguments]    ${operation_type}      ${timeout}=600s
      # Determine the log text based on the training type
    Run Keyword If    '${operation_type}' == 'ASR'       Set Test Variable    ${label_extension}    ASR任务完成, 查看终端进行下一步
...               ELSE IF    '${operation_type}' == 'One-click'    Set Test Variable    ${label_extension}    One-click formatting output
...               ELSE IF    '${operation_type}' == 'SoVITS'       Set Test Variable    ${label_extension}    ${operation_type} training output log
...               ELSE IF    '${operation_type}' == 'GPT'          Set Test Variable    ${label_extension}    ${operation_type} training output log
...               ELSE                                            Set Test Variable    ${label_extension}    ${None}


    Run Keyword If    '${operation_type}' == 'ASR'       Set Test Variable    ${operation_log_text}    ASR output log
...               ELSE IF    '${operation_type}' == 'One-click'    Set Test Variable    ${operation_log_text}    一键三连进程结束一键三连进程结束
...               ELSE IF    '${operation_type}' == 'SoVITS'       Set Test Variable    ${operation_log_text}    ${operation_type}训练完成
...               ELSE IF    '${operation_type}' == 'GPT'          Set Test Variable    ${operation_log_text}    ${operation_type}训练完成
...               ELSE                                            Set Test Variable    ${operation_log_text}    ${None}


    Wait Until Keyword Succeeds    ${timeout}    5s    Check Log   ${operation_type}    ${label_extension}      ${operation_log_text}
    
