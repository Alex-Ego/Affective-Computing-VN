################################################################################
## Neural Network Initialization
################################################################################

init offset = -3

init python:
    ################################################################################
    ## Perceptron Functions
    ################################################################################
    
    def readweights(txtfile):
        lines = renpy.file(txtfile).readlines()
        for i in range(len(lines)):
            lines[i] = float(lines[i])
        return lines
        
    def openfile(txtfile):
        lines = renpy.file(txtfile).readlines()
        i = 0
        for line in lines:
            if line[0] == "#":
                lines.pop(i)
            i = i + 1
        return lines
    
    def funcion_nucleo(pesos, entrada):
        arrayentradas = []
        for i in range(len(entrada)):
                arrayentradas.append(entrada[i])
        return list(zip(pesos, arrayentradas))

    def funcion_activacion(resultado_nucleo):
        fx = sum(x*y for x, y in resultado_nucleo)
        if fx >= 0:
            return 1
        else:
            return 0 
    
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
    
    def phraseanalysis(phrase, dictionary, pesos):
        criteria = [-1]
        mood = 0
        phrasewords = phrase.split(" ")
        for words in phrasewords:
            if words in dictionary:
                pass
            else:
                mood = mood + 1
        criteria.append(mood)
        criteria.append(len(phrase))
        try:
            criteria.append(mood/len(phrasewords))
        except:
            criteria.append(0)
        print(criteria)
        print(pesos)

        y=funcion_activacion(funcion_nucleo(pesos,criteria))
        if y == 0:
            renpy.say(e, "Sad phrase.")
        else:
            renpy.say(e, "Not a sad phrase.")
    
    ################################################################################
    ## Ren'py Interaction
    ################################################################################
    
    # Dictionary generation
    dictionary = {}
    referencewords = clean_dict(openfile("nndata/dictionary.txt"))
    addtodict(dictionary, referencewords)
    
    # Reading weights
    w = readweights("nndata/datafile.txt")
    
    
