

fw_100 = open('input_100.txt','w',encoding='utf-8')



with open('all_sentece.txt','r',encoding='utf-8') as  fr:
    lines = fr.readlines()
    for i in range(0,len(lines),2):
        if (i<100):
            fw_100.write(lines[i])
fw_100.flush()
fw_100.close()