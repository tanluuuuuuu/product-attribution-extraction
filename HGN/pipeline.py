import numpy as np
from typing import Optional, List, Tuple
import torch

### REFERENCES:
# - https://huggingface.co/transformers/v4.4.2/_modules/transformers/pipelines/token_classification.html

def ensure_tensor_on_device(**inputs):
    """
    Ensure PyTorch tensors are on the specified device.
    :param inputs:
    :return:
    """
    return {name: tensor.to('cuda') for name, tensor in inputs.items()}

def group_sub_entities(tokenizer, entities: List[dict]) -> dict:
    """
    Group together the adjacent tokens with the same entity predicted.

    Args:
        entities (:obj:`dict`): The entities predicted by the pipeline.
    """
    # Get the first entity in the entity group
    entity = entities[0]["entity"].split("-")[-1]
    scores = np.nanmean([entity["score"] for entity in entities])
    tokens = [entity["word"] for entity in entities]

    entity_group = {
        "entity_group": entity,
        "score": np.mean(scores),
        "word": tokenizer.convert_tokens_to_string(tokens),
        "start": entities[0]["start"],
        "end": entities[-1]["end"],
    }
    return entity_group

def group_entities(entities: List[dict], tokenizer, ignore_subwords) -> List[dict]:
    """
    Find and group together the adjacent tokens with the same entity predicted.

    Args:
        entities (:obj:`dict`): The entities predicted by the pipeline.
    """

    entity_groups = []
    entity_group_disagg = []

    if entities:
        last_idx = entities[-1]["index"]

    for entity in entities:

        is_last_idx = entity["index"] == last_idx
        is_subword = ignore_subwords and entity["is_subword"]
        if not entity_group_disagg:
            entity_group_disagg += [entity]
            if is_last_idx:
                entity_groups += [group_sub_entities(tokenizer, entity_group_disagg)]
            continue

        # If the current entity is similar and adjacent to the previous entity, append it to the disaggregated entity group
        # The split is meant to account for the "B" and "I" suffixes
        # Shouldn't merge if both entities are B-type
        if (
            (
                entity["entity"].split("-")[-1] == entity_group_disagg[-1]["entity"].split("-")[-1]
                and entity["entity"].split("-")[0] != "B"
            )
            and entity["index"] == entity_group_disagg[-1]["index"] + 1
        ) or is_subword:
            # Modify subword type to be previous_type
            if is_subword:
                entity["entity"] = entity_group_disagg[-1]["entity"].split("-")[-1]
                entity["score"] = np.nan  # set ignored scores to nan and use np.nanmean

            entity_group_disagg += [entity]
            # Group the entities at the last entity
            if is_last_idx:
                entity_groups += [group_sub_entities(tokenizer, entity_group_disagg)]
        # If the current entity is different from the previous entity, aggregate the disaggregated entity group
        else:
            entity_groups += [group_sub_entities(tokenizer, entity_group_disagg)]
            entity_group_disagg = [entity]
            # If it's the last entity, add it to the entity groups
            if is_last_idx:
                entity_groups += [group_sub_entities(tokenizer, entity_group_disagg)]

    return entity_groups

def postprocess(inputs, tokenizer, model, id2label):
    answers = []
    ignore_labels = ['O']
    grouped_entities = True
    ignore_subwords = False
    for sentence in inputs:
        # breakpoint()
        # Manage correct placement of the tensors
        tokens = tokenizer(
            sentence,
            return_attention_mask=False,
            return_tensors='pt',
            truncation=True,
            return_special_tokens_mask=True,
            return_offsets_mapping=tokenizer.is_fast,
            max_length = 512
        )
        if tokenizer.is_fast:
            offset_mapping = tokens.pop("offset_mapping").cpu().numpy()[0]
        else:
            offset_mapping = None
        
        special_tokens_mask = tokens.pop("special_tokens_mask").cpu().numpy()[0]
        
        # Forward
        with torch.no_grad():
            tokens = ensure_tensor_on_device(**tokens)
            entities = model(**tokens)['logits'].cpu().numpy()
            input_ids = tokens["input_ids"].cpu().numpy()[0]

        score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)
        labels_idx = score.argmax(axis=-1)

        entities = []
        # Filter to labels not in `self.ignore_labels`
        # Filter special_tokens
        filtered_labels_idx = [
            (idx, label_idx)
            for idx, label_idx in enumerate(labels_idx)
            if (id2label[label_idx] not in ignore_labels) and not special_tokens_mask[idx]
        ]

        for idx, label_idx in filtered_labels_idx:
            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]
                word_ref = sentence[start_ind:end_ind]
                word = tokenizer.convert_ids_to_tokens([int(input_ids[idx])])[0]
                is_subword = len(word_ref) != len(word)

                if int(input_ids[idx]) == tokenizer.unk_token_id:
                    word = word_ref
                    is_subword = False
            else:
                word = tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
                start_ind = None
                end_ind = None

            entity = {
                "word": word,
                "score": score[idx][label_idx].item(),
                "entity": id2label[label_idx],
                "index": idx,
                "start": start_ind,
                "end": end_ind,
            }

            if grouped_entities and ignore_subwords:
                entity["is_subword"] = is_subword

            entities += [entity]

        if grouped_entities:
            gr_entities = group_entities(entities, tokenizer, ignore_subwords)
            if len(gr_entities) > 0:
                answers += [gr_entities]
        # Append ungrouped entities
        else:
            answers += [entities]
    if len(answers) == 1:
        return answers[0]
    return answers