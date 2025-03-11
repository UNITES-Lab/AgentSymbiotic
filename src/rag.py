# rag-2part()

import argparse
import os
import re
import json
import tiktoken
from AgentOccam.retrieval import retrieval
from openai import AzureOpenAI


def extract_and_retrieve(input_string, source_list):
    match = re.search(r"\[External[^\]]+\]", input_string)
    if not match:
        print("No MATCH")
        # print(input_string)
        return []

    bracket_content = match.group(0)

    indices = [int(num) for num in re.findall(r"knowledge(\d+)", bracket_content)]

    result = [source_list[i] for i in indices if i < len(source_list)]

    return result


# from retrieval.run_retrieval import retrieval

client = AzureOpenAI(

)


def read_external_knowledge_from_log(log_dir, ids):
   
    external_knowledge_list = []  
    totaltokens = 0

    for idx, id in enumerate(ids, start=0):
        log_file_path = os.path.join(log_dir, f"{id}.log")

        if not os.path.exists(log_file_path):
            continue

       
        with open(log_file_path, "r", encoding="utf-8") as log_file:
            log_data = log_file.read()

        tokenizer = tiktoken.get_encoding("o200k_base")

        tokens = tokenizer.encode(log_data)

        totaltokens = totaltokens + len(tokens)
        if totaltokens < 50000:
            
            knowledge_dict = {f"External Knowledge{idx}": log_data}
            external_knowledge_list.append(knowledge_dict)
        else:
            break

    return json.dumps(external_knowledge_list)  





def get_examples(
    self,
    instruction,
    observation,
    in_file1="",
    in_file2="",
    interaction_history="",
    system_message="""You should give helpful, detailed, and polite responses to the user's queries.""",
    k_observation=2,  
    k_model=1,
    retriever="flat-contriever",
    log_dir="",
    log=False,
    filter=False,
    thinkprompt=True,
    call_func=None,
    sysprompt = "",
    mask = False,
):  
    print("---------------------------------------------------")
    if thinkprompt==True:
        print("--------------thinkprompt==True------------------------")
        system_message=sysprompt+"You should give helpful, detailed, and polite responses to the user's queries."
        # print(system_message)
    if mask==True:
        in_file1="output2.jsonl"
        in_file2="output.jsonl"
        log_dir="maskinstructions-7"


    args = argparse.Namespace(
        in_file=in_file1,
        outfile_prefix=None,
        cache_dir=None,
        retriever=retriever,
        granularity="session",
        index_expansion_method="none",
        index_expansion_llm="none",
        index_expansion_result_cache=None,
        index_expansion_result_join_mode="none",
        k=k_observation,
        query=observation,
    )
    id = retrieval(args)
    if log:
        print("--------------id---------------\n")
        print(id)
        print("--------------endid---------------\n")

    messages = [
        {
            "role": "user",
            "content": f"""Instruction: {instruction}
            Interaction History: {interaction_history}
            Observations: {observation}
            Please predict the Briefly summarize the significant steps or actions taken throughout the trajectory. And Provide a higher-level abstraction of the purpose or goal of the trajectory actions.""",
        }
    ]
    if log:
        print("--------------message---------------\n")
        print(messages)
        print("--------------endmessage---------------\n")

    if call_func:
        model_response = call_func(
                system_prompt=system_message, messages=messages
            )
    else:
        model_response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, max_tokens=4096, temperature=0.0
        )
        model_response = model_response.choices[0].message.content
    llm_prediction = model_response


    if log:
        print("--------------llm_prediction---------------\n")
        print(llm_prediction)
        print("--------------endllm_prediction---------------\n")

    args_old = args
    args = argparse.Namespace(
        in_file=in_file2,
        outfile_prefix=None,
        cache_dir=None,
        retriever=retriever,
        granularity="session",
        index_expansion_method="none",
        index_expansion_llm="none",
        index_expansion_result_cache=None,
        index_expansion_result_join_mode="none",
        k=k_model,
        query=llm_prediction,
    )
    id2 = retrieval(args)

    id.extend(id2)
    if log:
        print("--------------id12---------------\n")
        print(id)
        print("--------------end12---------------\n")
    external_knowledge = read_external_knowledge_from_log(log_dir, id)

    messages_ex = None
    llm_prediction_ex = None
    if filter == True:
        messages_ex = [
            {
                "role": "user",
                "content": f"""Instruction: {instruction}
                Interaction History: {interaction_history}
                Observations: {observation}
                External knowledge: {external_knowledge}
                Another gpt model needs to predict the next action based on the Instruction, Interaction History and Observations. The system message will tell it what it can do, instruction is our final goal, interaction history is some finished steps, observation is what we see now, and based on these thing we want gpt model to predict next action to achieve our goal. 
                Here is some action choice and meaning:
                click [id]: To click on an element with its numerical ID on the webpage. E.g., `click [7]` If clicking on a specific element doesn't trigger the transition to your desired web state, this is due to the element's lack of interactivity or GUI visibility. In such cases, move on to interact with OTHER similar or relevant elements INSTEAD.
                go_back: To return to the previously viewed page.
                stop [answer]: To stop interaction and return response. Present your answer within the brackets. If the task doesn't require a textual answer or appears insurmountable or finally find no answer(for example if there is no product processing or no place nearby, you can't choose a not correct answer), must indicate "N/A"! must indicate "N/A"! and additional reasons and all relevant information you gather as the answer. E.g., `stop [N/A ...]`. If return the direct response textual answer within brackets, The response should be the exact value/token without any additional description or explanation, E.g., For a token request, use stop [ABC_123] not stop [The token is ABC_123]. You don't need to do more exploration after finisded the task, just finished the task.
                note [content]: To take note of all important info w.r.t. completing the task to enable reviewing it later. E.g., `note [Spent $10 on 4/1/2024]`
                type [id] [content] [press_enter_after=0|1]: To type content into a field with a specific ID. By default, the "Enter" key is pressed after typing unless `press_enter_after` is set to 0. E.g., `type [15] [Carnegie Mellon University] [1]` If you can't find what you're looking for on your first attempt, consider refining your search keywords by breaking them down or trying related terms.
                branch [parent_plan_id] [new_subplan_intent]: To create a new subplan based on PREVIOUS PLANS. Ensure the new subplan is connected to the appropriate parent plan by using its ID. E.g., `branch [12] [Navigate to the "Issue" page to check all the issues.]`
                prune [resume_plan_id] [reason]: To return to a previous plan state when the current plan is deemed impractical. Enter the ID of the plan state you want to resume. E.g., `prune [5] [The current page lacks items "black speaker," prompting a return to the initial page to restart the item search.]`
                We additonally add External knowledge as a example to help finished the task. Please Evaluate whether the provided External knowledges can help another gpt model to predict a action, based on the current system message, Instruction, Interaction History, and Observations. Should focus on helpful not just releted. If any External knowledges are useful, identify which ones have value and explain why they are useful. Finally, output the useful External knowledges in the format [External knowledge0, External knowledge1].""",
            }
        ]
        if log:
            print("--------------message_ex---------------\n")
            print(messages_ex)
            print("--------------endmessage_ex---------------\n")

        if call_func:
            model_response = call_func(
                    system_prompt=system_message, messages=messages_ex
                )
        else:
            model_response = client.chat.completions.create(
                model="gpt-4o-mini", messages=messages_ex, max_tokens=4096, temperature=0.0
            )
            model_response = model_response.choices[0].message.content

        llm_prediction_ex = model_response

        if log:
            print("--------------llm_prediction_ex---------------\n")
            print(llm_prediction_ex)
            print("--------------llm_prediction_ex---------------\n")

        id = extract_and_retrieve(llm_prediction_ex, id)
        if log:
            print("--------------finalid---------------\n")
            print(id)
            print("--------------finalid---------------\n")
        if not id:
            external_knowledge = ""
        else:
            external_knowledge = read_external_knowledge_from_log(log_dir, id)

    return external_knowledge, {
        "id": id,
        "args": args,
        "args_old": args_old,
        "llm_prediction": llm_prediction,
        "external_knowledge": external_knowledge,
        "filter_input": messages_ex,
        "filter_output": llm_prediction_ex,
    }


if __name__ == "__main__":
    instruction = "Navigate to the target location."
    observation = "The target is located 3 miles northeast."

    result = get_examples(instruction=instruction, observation=observation, log=True)
    # print(result)
