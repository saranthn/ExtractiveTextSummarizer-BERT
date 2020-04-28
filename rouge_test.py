from rouge import Rouge

def rouge_score(gen_summary, gold_summary):
    gold_summary_formatted  = '.'.join(gold_summary)
    rouge = Rouge()
    scores = rouge.get_scores(gen_summary, gold_summary_formatted)
    return scores