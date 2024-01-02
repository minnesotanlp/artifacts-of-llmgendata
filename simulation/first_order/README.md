# Role Flipping

`role_flipping/find_flipping_interruption.py` explains how we detect the role-flipped and interruption messages in the conversations in the CAMEL AI-World dataset.

How to run:

1. Place raw dataset `ai_society_chat.zip` in the `role_flipping` folder (Dataset can be downloaded from https://huggingface.co/datasets/camel-ai/ai_society/tree/main)

2. Run `python role_flipping/find_flipping_interruption.py`

It produces 3 files:

1. `ai_society_chat.jsonl` Identical content as the raw dataset

2. `CAMEL_annotated.jsonl` This file contains 3 extra attributes than `ai_society_chat.jsonl`, they are:

   - *role_flipping_msg_indices*: a list of indices of role-flipped messages in the  conversation

   - *interruption_msg_indices*: a list of indices of interruption messages in the  conversation

   - *role_flipping_happens*: boolean true when *role_flipping_msg_indices* is not empty

     Please refer to the definition of *role-flipped messages* and *interruption messages* in the paper.

3. `CAMEL_annotated.csv` This file contains the same as `CAMEL_annotated.jsonl` but all messages are put under the *conversation* column. This file can be accessed at our [huggingface dataset card](https://huggingface.co/datasets/minnesotanlp/LLM-Artifacts/viewer/default/simulation_roleflip).

# Digression

