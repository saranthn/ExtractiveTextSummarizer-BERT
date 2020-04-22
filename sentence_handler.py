def sentence(content):
    start = "[CLS] "
    end = " [SEP]"
    f = open("error.txt","a")

    # Add [CLS] and [SEP] token at start and end of each sentence      
    sentence_count = len(content)    
    if sentence_count != 0:
        for i in range(sentence_count):
            content[i] = start + content[i]+ end
    else:
        f.write("error: " + str(j) + "\n")            
    f.close() 
    return content