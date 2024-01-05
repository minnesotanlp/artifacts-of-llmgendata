## Overview

Code and data used for finetuning a RoBERTa-base model from the first-order annotations and analyzing/visualizing the output.

## Data Directory

### `data/`

The train and val pkl files were used in the finetuning process. Only the gold labels in test files were used in the analyses/visualization steps.
Please contact authors for outputs from the finetuned model.

### Script Directory

## `scripts/`

- `run.sh`:
  - Description: Bash file used for running the finetuning code with different arguments (datasets, data order, etc.)
  - Usage:
    ```bash
    bash run.sh 
    ```

- `utils.py`:
  - Description: Contains functions shared between finetune_roberta.py and visualized.py. Imported into other files and not used on its own.

- `finetune_roberta.py`:
  - Description: Finetune RoBERTA-base model. Includes code from multi-headed RoBERTa class and custom trainer.
  - Usage: Example commands contains in run.sh

- `visualize.py`:
  - Description: Create visualizations from gold labels and the outputs from the finetuned RoBERTa-base model.
  - Usage:
    ```bash
    # make sure the if __name__=="__main__" block calls the expected functions (based on what outputs you want/have)
    python3 visualize.py 
    ```
