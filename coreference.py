import neuralcoref
import spacy

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

def coreference_handler(content):
    #convert list of sentences to paragraph
    combined_story = '. '.join(content)
    doc = nlp(combined_story)._.coref_resolved
    doc = nlp(doc)
    return [c.string.strip() for c in doc.sents if 600 > len(c.string.strip()) > 40]   