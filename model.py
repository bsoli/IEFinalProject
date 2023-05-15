from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import openai
import re
import time

from preprocess_data import get_relations_and_entities_as_string, get_data_instances
# reads in api key from text file
with open('openai_api_key.txt', 'r', encoding='utf8') as authkey:
    openai.api_key = authkey.readlines()[0]
    
task = 'task_3_description.txt'

with open(task, 'r', encoding='utf8') as td:
    task_prompt = ''.join(td.readlines())

model = 'gpt-3.5-turbo'

# used for training the model for task 3
training_mode = True
n_training_instances = 5

# used to convert gpt output to a python list 
def numbered_list_to_py_list(num_list_str: str) -> list[str]:
    num_list = re.compile(r'\d+\. .+', re.MULTILINE)
    if num_list.match(num_list_str):
        num_list_str = num_list.findall(num_list_str)
        num_list_str = '\n'.join(num_list_str)
        lines = num_list_str.strip().split('\n')
        items = [line.split('. ')[1] for line in lines]
        return items
    else:
        print(num_list_str)


# training mode creates a conversation where the model is shown data and given the correct classification, testing mode gives the model a training sample and returns its output
def chat_gpt_classifier(messages: list[dict], test_data: str = '', training_mode: bool=False, training_data: list[str]= None, gold_labels: list[str]=None):
    if training_mode:
        assert training_data and gold_labels
        for data, gold in zip(training_data, gold_labels):
            messages.append({'role': 'user', 'content': data})
            pred = openai.ChatCompletion.create(
                model=model,
                messages=messages, 
                temperature=.001, 
                max_tokens=100)
            gold_list = '\n'.join(gold)
            pred = pred['choices'][0]['message']['content']
            messages.append({'role': 'assistant', 'content': pred})
            messages.append({'role': 'user', 'content': f'Correct labels:\n{gold_list}'})
        return messages
    else:
        messages.append({'role': 'user', 'content': test_data})
        response = openai.ChatCompletion.create(
                model=model, 
                messages = messages,
                temperature=.001, 
                max_tokens=100)
        messages.pop()
        return numbered_list_to_py_list(response['choices'][0]['message']['content'])

if __name__ == '__main__':
    # read in dataset and process it 
    dataset = load_dataset("DFKI-SLT/SemEval2018_Task7")
    train = dataset['train']
    train_df = train.to_pandas()
    train_df['relation_and_ents'] = train_df.apply(lambda row: get_relations_and_entities_as_string(row), axis=1)
    train_df = train_df[train_df['relations'].apply(lambda x: len(x) > 0)]
    train_df, test_df = train_test_split(train_df, test_size=.2, shuffle=False)
    train_messages = [{'role': 'user', 'content': task_prompt}]

    # initiate training loop for few shot experiments
    if training_mode:
        training_data, labels = get_data_instances(train_df)
        training_data = training_data[:n_training_instances]
        labels = labels[:n_training_instances]
        train_messages = chat_gpt_classifier(messages=train_messages, training_mode=True, training_data=training_data, gold_labels=labels)
        print('Training Complete')
        #avoid data limits
        time.sleep(30)
    # initiates testing loop
    correct = []
    preds = [] 
    wait = 0
    test_data, test_labels = get_data_instances(test_df)
    for data, label in zip(test_data, test_labels):
        gpt_output = chat_gpt_classifier(train_messages, data, training_mode=False)
        # validates that gpt is not hallucinating extra labels or ignoring any entity pairs
        gpt_output = gpt_output if gpt_output else 'no output'
        if len(labels) != len(gpt_output) :
            print(gpt_output)
            print(labels)
        else:
            preds.extend(gpt_output)
            correct.extend(labels)
        # used to avoid data limits on api usage
        wait += 1
        if wait == 10:
            time.sleep(60)
            wait = 0
        
    
    print(classification_report(correct, preds))

