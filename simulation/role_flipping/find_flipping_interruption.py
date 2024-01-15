import json
import zipfile
from tqdm import tqdm
import pandas as pd


# 1. Convert the raw dataset to jsonl format
# Place the CAMEL AI-World dataset ai_society_chat.zip in the folder
# (Download from https://huggingface.co/datasets/camel-ai/ai_society/tree/main)
with zipfile.ZipFile("ai_society_chat.zip", "r") as z:
    filenames = z.namelist()

    with open("ai_society_chat.jsonl", "w") as f_out:
        for filename in tqdm(filenames, desc="Processing files"):
            with z.open(filename) as f_in:
                data = json.load(f_in)
                jsonl_str = json.dumps(data)
                f_out.write(jsonl_str + "\n")


# 2. Detect the role-flipping and interruption and annotate them
def check_flipping_interruption(chat):
    flipped_indices = []
    interrupted_indices = []
    for index, (key, value) in enumerate(chat.items()):
        if key.startswith("message_"):
            role_type = value.get("role_type", "")
            content = value.get("content", "")

            if role_type == "USER":
                if "Solution:" in content.strip():
                    flipped_indices.append(index - 4)
                elif (
                    not "Instruction:" in content.strip()
                    and not "<CAMEL_TASK_DONE>" in content.strip()
                ):
                    interrupted_indices.append(index - 4)
            elif role_type == "ASSISTANT":
                if "Instruction:" in content.strip():
                    flipped_indices.append(index - 4)
                elif (
                    not "Solution:" in content.strip()
                    and not "<CAMEL_TASK_DONE>" in content.strip()
                ):
                    interrupted_indices.append(index - 4)

    return flipped_indices, interrupted_indices


def annotate_flipping_and_interruption():
    chats = []

    with open("ai_society_chat.jsonl", "r") as f:
        for line in tqdm(f):
            data_dict = json.loads(line.strip())
            chats.append(data_dict)

    for chat in chats:
        flipped_indices, interrupted_indices = check_flipping_interruption(chat)
        chat["role_flipping_happens"] = len(flipped_indices) > 0
        chat["role_flipping_msg_indices"] = flipped_indices
        chat["interruption_msg_indices"] = interrupted_indices

    return chats


annotated_chats = annotate_flipping_and_interruption()


# 3. Save the annotated dataset
def dict_list_to_jsonl(dict_list, path):
    jsonl_str = "\n".join(json.dumps(dictionary) for dictionary in dict_list)
    with open(path, "w") as file:
        file.write(jsonl_str)


dict_list_to_jsonl(annotated_chats, "CAMEL_annotated.jsonl")


# 4. (Optional) Save the annotated dataset in csv format, the same as on the huggingface dataset card
# https://huggingface.co/datasets/minnesotanlp/LLM-Artifacts/viewer/default/simulation_roleflip
fixed_attributes = [
    "role_1",
    "role_2",
    "id",
    "original_task",
    "specified_task",
    "termination_reason",
    "num_messages",
    "role_flipping_happens",
    "role_flipping_msg_indices",
    "interruption_msg_indices",
]

data = []

with open("CAMEL_annotated.jsonl", "r") as file:
    for line in file:
        json_line = json.loads(line)

        row_data = {attr: json_line.get(attr, None) for attr in fixed_attributes}
        other_attributes = {
            k: v for k, v in json_line.items() if k not in fixed_attributes
        }

        row_data["conversation"] = json.dumps(other_attributes)

        data.append(row_data)

df = pd.DataFrame(data)

df.to_csv("CAMEL_annotated.csv", index=False)
