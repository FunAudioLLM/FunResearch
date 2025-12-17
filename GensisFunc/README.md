A multi-agent automated data production pipeline for generating tool-calling training data.

## Data
The relevant data is saved in the corresponding JSON files:
* `simple.json`: Use single tools, each called once, to complete the user's task. It can be either a single-turn or a multi-turn dialogue.
* `multiple.json`: Use multiple tools, each called once, to complete the user's task. It can be either a single-turn or a multi-turn dialogue.
* `parallel.json`: Use single tools, each called many times, to complete the user's task. It can be either a single-turn or a multi-turn dialogue.
* `parallel_multiple.json`: Use multiple tools, each called many times, to complete the user's task. It can be either a single-turn or a multi-turn dialogue.

## Generate Training Data

### Generate Dialogues
If you want to generate a single-turn dialogue in a single-task setting, run:
```bash
python generator.py input output mode
python generator.py --input simple.json --output dialogue.json --mode 1
```
`input`: The seed tool set you need to load，you can choose from: **simple.json**, **multiple.json**, **parallel.json**, **parallel_multiple.json**.

`output`：The file path where your generated dialogues will be saved.

`mode`：The type of dialogue you want to generate, you can choose from: **1(single-turn single task)**, **2(single-turn multi task)**, **3(multi-turn single task)**, **4(multi-turn multi task)**.

### Convert to Training Data
If you want to convert the generated dialogues into the corresponding training data, run:
```bash
python convert.py input output pool
python convert.py --input dialogue.json --output data.json --pool function.json
```
`input`: The path to the dialogues you generated.

`output`：The file path where your generated training data will be saved.

`pool`：The path to the tool pool you use, you can choose from: **simple.json**, **multiple.json**, **parallel.json**, **parallel_multiple.json**.

## Evaluation
[BFCL](https://github.com/ShishirPatil/gorilla?tab=readme-ov-file), [API-Bank](https://github.com/qiancheng0/ToolRL) and [ACEBench](https://github.com/ACEBench/ACEBench?tab=readme-ov-file) are used for evaluation.






