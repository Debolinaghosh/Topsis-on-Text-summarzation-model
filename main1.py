import pandas as pd
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer,BartForConditionalGeneration, BartTokenizer
                          
from rouge_score import rouge_scorer
from datasets import load_dataset

# Step 2: Load SAMSUM Dataset
dataset = load_dataset("samsum")
texts = dataset['train']['dialogue'][:10]  # Adjust the number of samples as needed
references = dataset['train']['summary'][:10]  # Use 'summary' field for reference summaries

# Step 3: Define Evaluation Function
def evaluate_model(model, tokenizer, texts, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    total_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}

    for i, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(**inputs)
        generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        reference_summary = references[i]

        scores = scorer.score(generated_summary, reference_summary)
        total_scores['rouge1'] += scores['rouge1'].fmeasure
        total_scores['rouge2'] += scores['rouge2'].fmeasure
        total_scores['rougeL'] += scores['rougeL'].fmeasure

    avg_scores = {key: value / len(texts) for key, value in total_scores.items()}
    return avg_scores

# Step 4: Evaluate Multiple Models
model_names = ["t5-base", "t5-small", "sshleifer/distilbart-cnn-12-6", 
               "facebook/bart-large-cnn", "facebook/bart-large-xsum",
               "facebook/bart-large-mnli"]

results = []

for model_name in model_names:
    if "t5" in model_name.lower():
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    elif "bart" in model_name.lower():
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)
    else:
        print(f"Unsupported model: {model_name}")
        continue

    scores = evaluate_model(model, tokenizer, texts, references)
    results.append({'Model': model_name, 'ROUGE-1': scores['rouge1'], 'ROUGE-2': scores['rouge2'], 'ROUGE-L': scores['rougeL']})
    print(f"Model: {model_name}, ROUGE-1: {scores['rouge1']}, ROUGE-2: {scores['rouge2']}, ROUGE-L: {scores['rougeL']}")

# Step 5: Save Results to CSV
df = pd.DataFrame(results)
df.to_csv('model_evaluation_results.csv', index=False)
