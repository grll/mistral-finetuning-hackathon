# Mistral Finetuning Hackathon 2024

If you are looking at how to run the solution: [see here](#-how-to-run)

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

### Leverage Fine-tuning for Dona

We decided to fine-tune our legal assitant agent Dona for several reason:

1. Dona is the front of our application dealing with client interactions. Since the input of the client is unrestricted, we need to make sure Dona is robust to prompt hacking and other strategies to distrupt her behavior. Our idea was to fine tune Dona on several prompt hacking scenario, making sure that she do not follow the bad instructions. So we created a dataset composed of a mixture of regular Dona reply on legit legal cases and placeholder reply on prompt hacking scenarios. The result is a Dona more resilient when the user goes off topic:

<img width="1186" alt="image" src="https://github.com/unit8co/mistral-hackathon-finetuning/assets/1738060/8ca57196-4841-4c9a-907f-e732a8d53a74">

2. Since we had to create this dataset of also real reply from Dona we also took the chance to distil from larger model usually providing more rich replies than mistral 7B smaller model. For that the dataset mention above and the replies from dona to real legal cases used to finetune Dona was made using GPT4-o. Our mistral 7B model finetuned is now inspired by GPT4-o replies when prividing a summary. We noticed a qualitive improvement of the result over raw mistral 7B outputs.

3. Dona is an autogen agent which means several round are expected until the final summary of your case is extracted from the conversation with her. It also means that it is the most costly part in term of LLM usage. Hence having a smaller model performing well on this part is both cost and performance (speed) beneficial if we were to scale the solution further.

### Leverage Fine-tuning for classifying 

We also leverage fine-tuning for the classification of the legal case among Civil, Public or Criminal law. Here we prepared a dataset containing legal cases and the category of law that applies. We first develop a baseline using traditional ML (TFIDF+LGBM), we then tried Mistral 7B by prompting the model only and finally Mistral 7B finetuned. While we noticed a significant increase in performance after finetuning mistral 7B on the classification task we were never able to match the performance of TFIDF+LGBM on this particular task.

Also after finetuning we observed that mistral 7B halllucinated a lot less law categories that our dataset didnt have and really stuck a lot more to Civil, Public or Criminal.

#### Classication Results (on Fold 0 of Stratified 5 Fold CV)

* TFIDF+LGBM: Accuracy 0.86
* Mistral 7B: Accuracy
* Mistral 7B - finetune: Accuracy
[@antoine include result]

## Limitations

* Support only Federal Laws at the moment
* Support only cases of Civil, Public or Criminal law
* Performance of our classifier on our training set was

## How to run

```bash
git clone git@github.com:unit8co/mistral-hackathon-finetuning.git
cd mistral-hackathon-finetuning

# make sure you have python 3.11+ not tested for other version
# you also need node + npm (tested with node v22.1.0, npm 10.7.0) to run the frontend

# few assets are first needed to be install and unzip [@antoine]
# just chroma db or something else @antoine ?

# we recommend creating a virtual env
python -m  venv .venv

# then install all the dependencies
pip install -r requirements.txt

# create a .env and enter your mistral API key
cp .env.template .env

# you can then start the backend with
PYTHONPATH=(pwd) python src/backend/main.py

# in another terminal window cd into the frontend folder of the repository and run the frontend
cd src/frontend
# install node dependencies
npm i
# run the frontend
npm run dev

# you now follow the localhost url display and start chatting with the Dona and Rachel.
```

