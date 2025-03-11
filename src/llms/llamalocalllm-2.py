import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import json
import time
from tqdm import tqdm  
from litellm import batch_completion, completion
import json
import re


os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""
# DEFAULT_SYSTEM_PROMPT = """You are an AI assistant performing tasks on a web browser. You will be provided with task objective and web page observations. You need to issue an action for this step.
# You are ONLY allowed to use the following action commands. Strictly adheres to the given format. Only issue one single action. One action!!
# {click [id]: To click on an element with its numerical ID on the webpage. E.g., `click [7]`  ;
# type [id] [content] [press_enter_after=0|1]: To type content into a field with a specific ID. By default, the "Enter" key is pressed after typing unless `press_enter_after` is set to 0. E.g., `type [15] [Carnegie Mellon University] [1]` .;
# stop [answer]: To stop interaction and return response. Present your answer within the brackets. If the task doesn't require a textual answer or appears insurmountable, additional reasons and all relevant information you gather as the answer. E.g., `stop [N/A ...]`. ;
# note [content]: To take note of all important info w.r.t. completing the task to enable reviewing it later. E.g., `note [Spent $10 on 4/1/2024]`;
# go_back: To return to the previously viewed page."""


DEFAULT_SYSTEM_PROMPT = """Please predict the action."""
DEFAULT_SYSTEM_PROMPT2 = """You are an AI assistant performing tasks on a web browser. You will be provided with task objective, current step, web page observations, previous plans, and interaction history. You need to issue an action for this step.You will have 20 steps in total to finish this task. If you are in the 20th step, use the stop action.

Generate the response in the following format:
INTERACTION HISTORY SUMMARY:
Emphasize all important details in the INTERACTION HISTORY section.
OBSERVATION DESCRIPTION:
Describe information in the CURRENT OBSERVATION section. Emphasize elements and features that are relevant or potentially helpful for fulfilling the objective in detail.
OBSERVATION HIGHLIGHT:
List the numerical ids of elements on the current webpage based on which you would issue your action. Also include elements on the current webpage you would attend to if you fail in the future and have to restore to this step. Don't include elements from the previous pages. Select elements at a higher hierarchical level if most their children nodes are considered crucial. Sort by relevance and potential values from high to low, and separate the ids with commas. E.g., `1321, 52, 756, 838`.
REASON:
Provide your rationale for proposing the subsequent action commands here.
ACTION CANDIDATE:
Propose ALL potential good actions at this step. even if there is only one action, it must follow this format. Itemize the actions using this format: `- reason: [{reason_for_proposing_the_following_action0}]\n- action: [{action0_command}]\n\n- reason: [{reason_for_proposing_the_following_action1}]\n- action: [{action1_command}]\n\n...`.

You are ONLY allowed to use the following action commands. Strictly adheres to the given format. Only issue one single action.
If you think you should refine the plan, use the following actions:
- branch [parent_plan_id] [new_subplan_intent]: To create a new subplan based on PREVIOUS PLANS. Ensure the new subplan is connected to the appropriate parent plan by using its ID. E.g., `branch [12] [Navigate to the "Issue" page to check all the issues.]`
- prune [resume_plan_id] [reason]: To return to a previous plan state when the current plan is deemed impractical. Enter the ID of the plan state you want to resume. E.g., `prune [5] [The current page lacks items "black speaker," prompting a return to the initial page to restart the item search.]`
Otherwise, use the following actions:
- click [id]: To click on an element with its numerical ID on the webpage. E.g., `click [7]` If clicking on a specific element doesn't trigger the transition to your desired web state, this is due to the element's lack of interactivity or GUI visibility. In such cases, move on to interact with OTHER similar or relevant elements INSTEAD.
- type [id] [content] [press_enter_after=0|1]: To type content into a field with a specific ID. By default, the "Enter" key is pressed after typing unless `press_enter_after` is set to 0. E.g., `type [15] [Carnegie Mellon University] [1]` If you can't find what you're looking for on your first attempt, consider refining your search keywords by breaking them down or trying related terms.
- stop [answer]: To stop interaction and return response. Present your answer within the brackets. If the task doesn't require a textual answer or appears insurmountable or finally find no answer(for example if there is no product processing or no place nearby, you can't choose a not correct answer), must indicate "N/A"! must indicate "N/A"! and additional reasons and all relevant information you gather as the answer. E.g., `stop [N/A ...]`. If return the direct response textual answer within brackets, The response should be the exact value/token without any additional description or explanation, E.g., For a token request, use stop [ABC_123] not stop [The token is ABC_123]. You don't need to do more exploration after finisded the task, just finished the task.
- note [content]: To take note of all important info w.r.t. completing the task to enable reviewing it later. E.g., `note [Spent $10 on 4/1/2024]`
- go_back: To return to the previously viewed page."""


MODEL_PATH = ""
print(MODEL_PATH)
# Load tokenizer and model
print("Loading LLaMA model. This might take a while...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

# basemodel = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
# model = PeftModel.from_pretrained(basemodel, "")

# Ensure `pad_token` is set to avoid issues with padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully!")



def parse_reasons_and_actions(input_string):
            # pattern = r"- reason: \[(.*?)\]\s*(?:- action: \[(.*?)\])?\s*(?:\n|\Z)"
            pattern = r"(?:ACTION CANDIDATES:\s*)?- reason: (?:\[)?(.*?)(?:\])?\s*- action: \[?(.*?(?:\[.*?\])*)\]?(?=(?:\s*- reason:|\s*\Z))"

            matches = re.findall(pattern, input_string, re.DOTALL)

            parsed_data = []
            for match in matches:
                reason = match[0].strip()
                action = match[1].strip()
                if reason and action:
                    parsed_data.append({"reason": reason, "action": action})

            return parsed_data







def extract_action_text(observation, action):

  
    action = re.sub(r"(type|click)\s+\[", r"\1[", action) 
    action = re.sub(r"\]\s+\[", "][", action) 


    match = re.search(r"\[(\d+)]", action)
    if not match:
        return action  


    placeholder = match.group(1)  


    search_pattern = rf"{placeholder}[^\t]*" 
    result = re.search(search_pattern, observation)

    if result:

        replacement = result.group(0)  
        action = re.sub(r"\[\d+]", f"[{replacement}]", action, count=1)

    return action


def query_action(url, objective, action, metadata_file):
 
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

 
    candidates = [
        entry for entry in metadata
        if entry["objective"] == objective and entry["url"] == url
    ]

    if not candidates:
        return 0  
   
    for candidate in candidates:
        file_name = candidate["file_name"]
        step_index = candidate["step_index"]

      
        with open(file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
            step = data["trajectory"][step_index]

            if "action_candidates" in step:
               
                actions_to_compare = [
                    extract_action_text(step.get("observation", ""), candidate_action["action"])
                    for candidate_action in step["action_candidates"]
                ]
            else:
            
                actions_to_compare = [
                    extract_action_text(step.get("observation", ""), step.get("action", ""))
                ]

            
            processed_action = extract_action_text(step.get("observation", ""), action)
            if processed_action in actions_to_compare:
                return 1  

    return 0 





def process_string_with_reasons(input_string, target_action):

  
    

   
    parsed_candidates = parse_reasons_and_actions(input_string)

    if not parsed_candidates:
        print("----------can't get reason/action -----------------\n\n")
        return input_string
    print("----------parsed_candidates -----------------\n\n")
    print(parsed_candidates)
    
    for candidate in parsed_candidates:
        
        reason = candidate["reason"]
        action = candidate["action"]

        if target_action == action:        
            return "" 

    # no same action，choose first reason-action 
    first_candidate = parsed_candidates[0]
    replacement_content = f"REASON: {first_candidate['reason']}\n\nACTION: {first_candidate['action']}"

   
    updated_string = re.sub(r"ACTION CANDIDATE:.*", replacement_content, input_string, flags=re.DOTALL)

    return updated_string


def arrange_message_for_llamalocal(item_list):
    for item in item_list:
        if item[0] == "image":
            raise NotImplementedError()
    prompt = "".join([item[1] for item in item_list])
    return prompt

def call_llamalocal_with_messages(messages, model_id="meta.llama3-8b-instruct-v1:0", system_prompt=DEFAULT_SYSTEM_PROMPT,url="",objective="objective"):
    return call_llamalocal(prompt=messages, model_id=model_id, system_prompt=system_prompt,url=url,objective=objective)

def call_llamalocal(
    prompt,
    model_id="llama3-13b",
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    url="",objective=""
):  
    action = None
    if url=="" or objective=="":
        print("------------no url or objective ----------------\n\n")
    # url = url.replace
    
    print("-----------url-------------\n\n")
    print(url)
    print("-----------objective-------------\n\n")
    print(objective)
    # print("-------system_prompt----------")
    # print(system_prompt)
    # print("-------endsystem_prompt----------")
    # system_prompt=DEFAULT_SYSTEM_PROMPT,
    """
    Use the LLaMA3 model to generate a response based on the given prompt, truncating the input to a maximum of 4000 tokens if necessary.

    Parameters:
        prompt (str): The user query or input.
        model_id (str): The identifier for the LLaMA model to use (default: llama3-13b).
        system_prompt (str): System-level instructions to guide the assistant's behavior.

    Returns:
        str: The generated response from the LLaMA model.
    """
    # Combine the system prompt with the user input
    full_prompt = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    f"### Instruction:\n{DEFAULT_SYSTEM_PROMPT}\n\n"
    f"### Input:\n{prompt}\n\n"
    "### Response:"
)
    # .replace("INTERACTION HISTORY SUMMARY","INTERACTION HISTORY ")
    print("-------full_prompt----------\n")
    print(full_prompt)
    print("-------endfull_prompt----------\n")
    
    pattern = r"INTERACTION HISTORY.*?(INTERACTION HISTORY SUMMARY|CURRENT OBSERVATION):"
    full_prompt = re.sub(pattern, r"\1:",full_prompt, flags=re.DOTALL).replace("INTERACTION HISTORY SUMMARY","INTERACTION HISTORY ")
    print("-------full_prompt2----------\n")
    print(full_prompt)
    print("-------endfull_prompt2----------\n")


    
    # Tokenize the input while enabling truncation and padding
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",  # Return PyTorch tensors
        truncation=True,      # Truncate input to the max length
        max_length=9971,      # Set maximum input length
        padding=True          # Enable padding for consistent input shapes
    )
    total_tokens = len(inputs)
    if not total_tokens>13000:
        # Move input tensors to the same device as the model
        device = model.device  # Get the model's device (e.g., cuda:0)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate a response using the model
        print("Generating response...")
        outputs = model.generate(
            inputs["input_ids"],           # Input token IDs
            attention_mask=inputs["attention_mask"],  # Attention mask for padding
            max_new_tokens=1024,          # Limit the number of tokens in the output
            temperature=0.7,              # Sampling temperature for diversity
            top_p=0.9,                    # Nucleus sampling for token selection
            do_sample=True                # Enable sampling (vs. greedy decoding)
        )

        # Decode the output tokens into a human-readable string
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # .replace("### Response:", "### Response: ACTION:")
        # if "### Response:" in response:
        #     response = response.split("### Response:")[1].strip()
        # else:
        #     print("no'### Response:' 。")
            
        
        print("-------llama response----------\n")
        print(response)
        print("-------llama response----------\n") 
        
            
        match = re.search(r"ACTION:\s*(.+)", response, re.DOTALL)
        if match:
            
            action = match.group(1).replace("\n", "").strip()
            print("-----------get action-------------\n\n")
            print(action)

        else:
            print("------------cant get action----------------\n\n")
    
    if not action:
        action="stop [no action]"
    result = query_action(url, objective, action,  "metadata.json")
    
    print("-----------result-------------\n\n")
    print(result)
    
    if result == 0:
        print("------------try llm----------------\n\n")
        full_promptllm = (
            f"{DEFAULT_SYSTEM_PROMPT2}\n\n"
            f"### Input:\n{prompt}\n\n"
            "### Response:"
        ) 
        responsellm = completion(
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            messages=[{"content": full_promptllm, "role": "user"}],
        )
        # print(responsellm)
        responsellm = responsellm.choices[0].message.content  
              
        llmresponse = process_string_with_reasons(responsellm,action)
        if not llmresponse:
            print("-----------------positive llm - local action -use local -----------------\n\n")
            
        else:
            print("-----------------negetive llm - local action -use llm-----------------\n\n")
            response = llmresponse
    else:
        print("-----------find local positive pair-----------\n")

    # if full_prompt in response:
    #     response = response[len(full_prompt):].strip()
    print("----------response----------\n")
    print(response)
    print("----------endresponse----------\n")
    return response








if __name__ == "__main__":
    # Example usage of the function
    

    try:
        result = call_llamalocal(
            """OBJECTIVE:
What is the total count of Pending reviews amongst all the reviews?
CURRENT OBSERVATION:
RootWebArea 'Dashboard / Magento Admin'
        link [178] 'Magento Admin Panel'
        menubar [85]
                link [87] 'DASHBOARD'
                link [90] 'SALES'
                link [96] 'CATALOG'
                link [102] 'CUSTOMERS'
                link [108] 'MARKETING'
                link [114] 'CONTENT'
                link [120] 'REPORTS'
                link [138] 'STORES'
                link [144] 'SYSTEM'
                link [150] 'FIND PARTNERS & EXTENSIONS'
        heading 'Dashboard'
        link [254] 'admin'
        link [256]
        textbox [894] [required: False]
        main
                text 'Scope:'
                button [262] 'All Store Views'
                link [265] 'What is this?'
                button [240] 'Reload Data'
                HeaderAsNonLandmark [898] 'Advanced Reporting'
                text "Gain new insights and take command of your business' performance, using our dynamic product, order, and customer reports tailored to your customer data."
                link [902] 'Go to Advanced Reporting'
                text 'Chart is disabled. To enable the chart, click'
                link [906] 'here'
                text 'Revenue'
                text 'Tax'
                text 'Shipping'
                text 'Quantity'
                tablist [57]
                        tab [59] 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers' [selected: True]
                                link [67] 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers'
                                        generic [952] 'The information in this tab has been changed.'
                                        generic [953] 'This tab contains invalid data. Please resolve this before saving.'
                                        generic [954] 'Loading...'
                        tab [61] 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Most Viewed Products' [selected: False]
                                link [69] 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Most Viewed Products'
                                        generic [1049] 'The information in this tab has been changed.'
                                        generic [1050] 'This tab contains invalid data. Please resolve this before saving.'
                                        generic [1051] 'Loading...'
                        tab [63] 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... New Customers' [selected: False]
                                link [71] 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... New Customers'
                                        generic [1055] 'The information in this tab has been changed.'
                                        generic [1056] 'This tab contains invalid data. Please resolve this before saving.'
                                        generic [1057] 'Loading...'
                        tab [65] 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Customers' [selected: False]
                                link [73] 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Customers'
                                        generic [1061] 'The information in this tab has been changed.'
                                        generic [1062] 'This tab contains invalid data. Please resolve this before saving.'
                                        generic [1063] 'Loading...'
                tabpanel 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers'
                        table
                                row '| Product | Price | Quantity |'
                                row '| --- | --- | --- |'
                                row '| Sprite Stasis Ball 65 cm | 27.00 | 6 |'
                                row '| Quest Lumaflex Band | 19.00 | 6 |'
                                row '| Sprite Yoga Strap 6 foot | 14.00 | 6 |'
                                row '| Sprite Stasis Ball 55 cm | 23.00 | 5 |'
                                row '| Overnight Duffle | 45.00 | 5 |'
                text 'Lifetime Sales'
                text 'Average Order'
                text 'Last Orders'
                table
                        row '| Customer | Items | Total |'
                        row '| --- | --- | --- |'
                        row '| Sarah Miller | 5 | 194.40 |'
                        row '| Grace Nguyen | 4 | 190.00 |'
                        row '| Matt Baker | 3 | 151.40 |'
                        row '| Lily Potter | 4 | 188.20 |'
                        row '| Ava Brown | 2 | 83.40 |'
                text 'Last Search Terms'
                table
                        row '| Search Term | Results | Uses |'
                        row '| --- | --- | --- |'
                        row '| tanks | 23 | 1 |'
                        row '| nike | 0 | 3 |'
                        row '| Joust Bag | 10 | 4 |'
                        row '| hollister | 1 | 19 |'
                        row '| Antonia Racer Tank | 23 | 2 |'
                text 'Top Search Terms'
                table
                        row '| Search Term | Results | Uses |'
                        row '| --- | --- | --- |'
                        row '| hollister | 1 | 19 |'
                        row '| Joust Bag | 10 | 4 |'
                        row '| Antonia Racer Tank | 23 | 2 |'
                        row '| tanks | 23 | 1 |'
                        row '| WP10 | 1 | 1 |'
        contentinfo
                link [244]
                text 'Copyright 2024 Magento Commerce Inc. All rights reserved.'
                text 'ver. 2.4.6'
                link [247] 'Privacy Policy'
                link [249] 'Account Activity'
                link [251] 'Report an Issue'"""
        )
        print(result)
    except ValueError as e:
        print(f"Error: {e}")
        
        
        
        
        
        
        
