# Mistral Finetuning Hackathon 2024

## Alplex an AI based virtual lawyer office

We introduce Alplex an AI based virtual lawyer office that helps you tackle your legal issues grounded on swiss laws.

Upon receiving your legal case we first offer you the chance to clarify and summarize it with our AI Legal Assistant Dona (an autogen Agent backed by a finetuned mistral 7B).

Once you are happy with your case our second AI para-legal agent Rachel take over to classify your case in the right law category and perform a RAG over relevant swiss laws (backed by mistral-large model).

This is what the application look like:

![image](https://github.com/unit8co/mistral-hackathon-finetuning/assets/1738060/6817ec8a-19bf-4cfb-9484-f42ae4ffd175)

We leveraged mistral finetuning API for 2 key parts of our solution:

1. improve several aspects of Dona including guardrailing and distilling from larger models.
2. improve the classification of law cases into relevant law categories.

Details of how finetuning improved the solution is provided below.

## Solution Diagram

![image](https://github.com/unit8co/mistral-hackathon-finetuning/assets/1738060/75e9bf20-567d-40b9-b81e-22064b63f26b)


## Details of our contributions

### Leverage Fine-tuning for guardrailing


### Leverage Fine-tuning for classifying 

## Limitations

* Support only Federal Laws at the moment
* Support only cases of Civil, Public or Criminal law
* Performance of our classifier on our training set was 
