from rouge import Rouge

def rouge_score(gen_summary, gold_summary):
    gold_summary_formatted  = '.'.join(gold_summary)
    rouge = Rouge()
    scores = rouge.get_scores(gen_summary, gold_summary_formatted)
    R1_p = scores[0]["rouge-1"]["p"]
    R1_r = scores[0]["rouge-1"]["r"]
    R1_f = scores[0]["rouge-1"]["f"]

    R2_p = scores[0]["rouge-2"]["p"]
    R2_r = scores[0]["rouge-2"]["r"]
    R2_f = scores[0]["rouge-2"]["f"]

    RL_p = scores[0]["rouge-l"]["p"]
    RL_r = scores[0]["rouge-l"]["r"]
    RL_f = scores[0]["rouge-l"]["f"]
    return R1_p, R1_r, R1_f, R2_p, R2_r, R2_f, RL_p, RL_r, RL_f