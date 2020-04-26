from rouge import Rouge

def rouge_score(gen_summary, gold_summary):
    gold_summary_formatted = ""
    for i in range(len(gold_summary)):
        gold_summary_formatted += "[CLS] " + gold_summary[i] + " [SEP]. " 
    rouge = Rouge()
    print("##################")
    print(gold_summary_formatted)
    print("##################")
    print(gen_summary)
    print("##################")
    scores = rouge.get_scores(gen_summary, gold_summary_formatted)
    return scores