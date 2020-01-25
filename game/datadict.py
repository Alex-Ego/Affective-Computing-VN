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
        if line[len(line)-1] == "\n":
            line.pop(len(line)-1)
        if line != "":
            line = ''.join(line)
            line = str(line)
            cleanlines.append(line)

    #Making the info useful
    cleandata = []
    for lines in cleanlines:
        lines = lines.split("\t")
        lines[0] = lines[0].split(" ")
        for word in lines[0]:
            i = 0
            if word.isalnum() == False:
                lines[0].pop(i)
            i = i + 1
        cleandata.append(lines)
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
    for feels in data:
        for word in feels[0]:
            d[word] = feels[1]
