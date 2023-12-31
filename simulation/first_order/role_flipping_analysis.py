# 1. Convert the raw dataset
# Place the CAMEL AI-World dataset ai_society_chat.zip in the folder
# (Download from https://huggingface.co/datasets/camel-ai/ai_society/tree/main)
import json
import zipfile
from tqdm import tqdm

with zipfile.ZipFile("ai_society_chat.zip", "r") as z:
    filenames = z.namelist()

    with open("ai_society_chat.jsonl", "w") as f_out:
        for filename in tqdm(filenames, desc="Processing files"):
            with z.open(filename) as f_in:
                data = json.load(f_in)
                jsonl_str = json.dumps(data)
                f_out.write(jsonl_str + "\n")


# 2. Detect the role-flipping and interruption
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
