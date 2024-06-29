# Mistral Finetuning Hackathon 2024

For instructions on running the solution, [click here](#how-to-run).

## Alplex: An AI-based Virtual Law Office

Introducing Alplex, an AI-powered virtual law office designed to assist you with legal issues based on Swiss laws.

### Key Features

1. **AI Legal Assistant - Dona**:
   - **Clarification & Summarization**: Receive your case and help summarize it.
   - **Technology**: Powered by a fine-tuned Mistral 7B model.
   
2. **AI Paralegal - Rachel**:
   - **Case Classification**: Classifies your case into the correct legal category.
   - **RAG over Swiss Laws**: Uses a large Mistral model to perform Retrieval-Augmented Generation over relevant Swiss laws.

### Application Interface

![Application Interface](https://github.com/unit8co/mistral-hackathon-finetuning/assets/1738060/6817ec8a-19bf-4cfb-9484-f42ae4ffd175)

### Fine-tuning with Mistral API

We leveraged the Mistral fine-tuning API for two critical aspects:

1. **Improving Dona**: Enhanced guardrails and distilled from larger models.
2. **Better Case Classification**: Optimized classification accuracy for legal cases.

### Solution Diagram

![Solution Diagram](https://github.com/unit8co/mistral-hackathon-finetuning/assets/1738060/75e9bf20-567d-40b9-b81e-22064b63f26b)

## Finetuning Usage

### Fine-tuning for Dona

#### Goals

1. **Robust Client Interaction**:
   - Ensured resilience against prompt hacking.
   - Created a dataset with a mix of legitimate replies and placeholders for prompt hacking scenarios.

2. **Enhanced Responses**:
   - Distilled from larger models to improve response quality.
   - Used GPT-4 outputs to inspire the Mistral 7B model for better summaries.

3. **Cost and Performance Efficiency**:
   - Autogen agent requiring multiple interactions.
   - Fine-tuned smaller model for efficiency and scalability.

![Fine-tuning Results](https://github.com/unit8co/mistral-hackathon-finetuning/assets/1738060/8ca57196-4841-4c9a-907f-e732a8d53a74)

### Fine-tuning for Classification

We prepared a dataset of legal cases categorized under Civil, Public, or Criminal law and evaluated various models:

1. **Baseline**: Traditional ML (TFIDF+LGBM).
2. **Mistral 7B**: Prompting only.
3. **Mistral 7B (Fine-tuned)**: Significant performance improvement, reduced hallucinations.

#### Classification Results (Fold 0 of Stratified 5-Fold CV)

* TFIDF+LGBM: Accuracy 0.86
* Mistral 7B: Accuracy [Result needed]
* Mistral 7B (Fine-tuned): Accuracy [Result needed]

## Limitations

* Supports only Federal Laws.
* Handles only Civil, Public, or Criminal law cases.
* Performance on our training set needs improvement.

## How to Run

```bash
git clone git@github.com:unit8co/mistral-hackathon-finetuning.git
cd mistral-hackathon-finetuning

# Ensure you have Python 3.11+ and Node.js + npm (tested with Node v22.1.0, npm 10.7.0) for the frontend.

# Install necessary assets [Details required]

# Create a virtual environment
python -m venv .venv

# Install dependencies
pip install -r requirements.txt

# Create a .env file and enter your Mistral API key
cp .env.template .env

# Start the backend
PYTHONPATH=$(pwd) python src/backend/main.py

# In another terminal, navigate to the frontend folder and run the frontend
cd src/frontend
# Install Node.js dependencies
npm install
# Run the frontend
npm run dev

# Follow the localhost URL displayed to start interacting with Dona and Rachel.
```
