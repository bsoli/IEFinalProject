
import pandas as pd
import numpy as np



relation_decoder = {0: 'NONE', 1: 'USAGE', 2: 'RESULT', 3: 'MODEL-FEATURE', 4: 'PART_WHOLE', 5: 'TOPIC', 6: 'COMPARE' }

# gets the string representation for the entities from the character index representation
def get_relations_and_entities_as_string(row):
    relations_and_ents = []
    ents = dict()
    for ent in row.entities:
        ents[ent['id']] = row.abstract[ent['char_start']:ent['char_end']]
    for rel in row.relations:
        relations_and_ents.append((relation_decoder[rel['label']], ents[rel['arg1']], ents[rel['arg2']]))
    return relations_and_ents

# creates lists of training instances and labels from the dataframe.
def get_data_instances(df: pd.DataFrame):
    abstracts = df.abstract.tolist()
    entity_relations = df.relation_and_ents.tolist()
    train_data = []
    labels = []
    for abstract, ent_rel in zip(abstracts, entity_relations):
        entities = []
        entity_labels = []
        for i, rel in enumerate(ent_rel, start=1):
            label, arg1, arg2 = rel
            entity_labels.append(f'{i}. {label}')
            entities.append(f'{i}. {arg1} and {arg2}')
        labels.append(entity_labels)
        entity_list = '\n'.join(entities)
        data = f"""Abstract: {abstract}
                Entities:\n{entity_list}"""
        train_data.append(data)
    return (train_data, labels)