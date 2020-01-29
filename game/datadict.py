

def filterpunct(text):
    punctuation = [".", ",", ":", "\"", "!", "?", "*", "-", "+", "(", ")", "$", "%", "", "~", "#"]
    if(text in punctuation):
        return False
    else:
        return True

def openfile(txtfile):
    txt = open(txtfile, "r")
    lines = txt.readlines()
    i = 0
    for line in lines:
        if line[0] == "#":
            lines.pop(i)
        i = i + 1
    return lines

def clean_data(data):
    #Cleaning the formatting
    cleanlines = []
    for line in data:
        line = list(line)
        filterline = filter(filterpunct, line)
        line = list(filterline)
        if line[len(line)-1] == "\n":
            line.pop(len(line)-1)
        if line != '':
            line = ''.join(line)
            line = str(line)
            cleanlines.append(line)
        

    #Making the info useful
    cleandata = []
    for lines in cleanlines:
        lines = lines.split("\t")
        lines[0] = lines[0].split(" ")
#        for word in lines[0]:
#            i = 0
#            if word.isalnum() == False:
#                lines[0].pop(i)
#            i = i + 1
        cleandata.append(lines)
        #print(cleandata)
    return cleandata

def clean_dict(d):
    #Cleaning the formatting
    cleanlines = []
    for line in d:
        line = list(line)
        if line[len(line)-1] == "\n":
            line.pop(len(line)-1)
        if line != '':
            line = ''.join(line)
            line = str(line)
            cleanlines.append(line)

    #Making the info useful
    cleandata = []
    for lines in cleanlines:
        lines = lines.split("\t")
        lines[0] = lines[0].split(", ")
        cleandata.append(lines)
    return cleandata

def addtodict(d, data):
    '''Adds the data to the dictionary d'''
    for feels in data:
        for word in feels[0]:
            d[word] = feels[1]

def datadump(data_source):
    '''Makes the whole process of dumping the text in data_source to a vector automatically'''
    return clean_data((openfile(data_source)))


