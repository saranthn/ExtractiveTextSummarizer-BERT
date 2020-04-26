from sumeval.metrics.rouge import RougeCalculator

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

    rouge = RougeCalculator(stopwords=True, lang="en")

    rouge_1 = rouge.rouge_n(
                summary=gen_summary,
                references=gold_summary_formatted,
                n=1)

    rouge_2 = rouge.rouge_n(
                summary=gen_summary,
                references=gold_summary_formatted,
                n=2)

    rouge_l = rouge.rouge_l(
                summary=gen_summary,
                references=gold_summary_formatted)

    rouge_be = rouge.rouge_be(
                summary=gen_summary,
                references=gold_summary_formatted)

    scores={"ROUGE-1":rouge_1,"ROUGE-2":rouge_2,"ROUGE-L":rouge_l,"ROUGE-BE":rouge_be}
    return scores 
