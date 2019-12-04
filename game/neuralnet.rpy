# ################################################################################
# ## Neural Network Initialization
# ################################################################################

# init offset = -3

# init python:
    # from __future__ import absolute_import, division, print_function, unicode_literals

    # def openfile(txtfile):
       # lines = renpy.file(txtfile).readlines()
       # i = 0
       # for line in lines:
           # if line[0] == "#":
               # lines.pop(i)
           # i = i + 1
       # return lines

    # # def openfile(txtfile):
        # # txt = open(txtfile, "r")
        # # lines = txt.readlines()
        # # i = 0
        # # for line in lines:
            # # if line[0] == "#":
                # # lines.pop(i)
            # # i = i + 1
        # # return lines

    # def clean_data(data):
        # #Cleaning the formatting
        # cleanlines = []
        # for line in data:
            # line = list(line)
            # if line[len(line)-1] == "\n":
                # line.pop(len(line)-1)
            # if line != "":
                # line = ''.join(line)
                # line = str(line)
                # cleanlines.append(line)

        # #Making the info useful
        # cleandata = []
        # for lines in cleanlines:
            # lines = lines.split("\t")
            # lines[0] = lines[0].split(" ")
            # for word in lines[0]:
                # i = 0
                # if word.isalnum() == False:
                    # lines[0].pop(i)
                # i = i + 1
            # cleandata.append(lines)
        # return cleandata

    # def clean_dict(d):
        # #Cleaning the formatting
        # cleanlines = []
        # for line in d:
            # line = list(line)
            # if line[len(line)-1] == "\n":
                # line.pop(len(line)-1)
            # if line != '':
                # line = ''.join(line)
                # line = str(line)
                # cleanlines.append(line)

        # #Making the info useful
        # cleandata = []
        # for lines in cleanlines:
            # lines = lines.split("\t")
            # lines[0] = lines[0].split(", ")
            # cleandata.append(lines)
        # return cleandata

    # def funcion_nucleo(pesos, entrada):
        # arrayentradas = []
        # for i in range(len(entrada)):
                # arrayentradas.append(entrada[i])
        # return list(zip(pesos, arrayentradas))

    # def funcion_activacion(resultado_nucleo):
        # fx = sum(x*y for x, y in resultado_nucleo)
        # if fx >= 0:
            # return 1
        # else:
            # return 0 

    # def addtodict(d, data):
        # for feels in data:
            # for word in feels[0]:
                # d[word] = feels[1]

    # #Data cleaning and storing
    # testdata = clean_data((openfile("nndata/test/testdata.txt")))
    # traindata = clean_data((openfile("nndata/test/traindata.txt")))

    # #Dictionary generation
    # dictionary = {}
    # referencewords = clean_dict(openfile("nndata/dictionary.txt"))
    # addtodict(dictionary, referencewords)

    # print("==============================================================================================")
    # #print(dictionary)

    # #Dictionary vs test

    # entradas = []
    # for example in traindata:
        # data = []
        # criteria = [-1]
        # mood = 0
        # for word in example[0]:
            # answer = int(example[len(example)-1])
            # if word in dictionary:
                # mood = mood + int(dictionary[word])
        # criteria.append(mood)
        # criteria.append(len(example[0]))
        # try:
            # criteria.append(mood/len(example[0]))
        # except:
            # criteria.append(0)
        # data.append(criteria)
        # data.append(answer)
        # entradas.append(data)

    # #random.shuffle(entradas)

    # ##================================================================ Placeholder
    # max_iter=100
    # tasa_aprendizaje=0.000000062
    # pesos=[1,0,0,0]

    # iter=0
    # pincorrectos=1.0

    # while(pincorrectos>0.002 and iter<max_iter):

        # incorrectos=0
            
        # print("\nITERATION " + str(iter))

        # for entrada in entradas:
                # #print("Procesando entrada: ")
                # #print(entrada[0])
                # #print()
                # y=funcion_activacion(funcion_nucleo(pesos,entrada[0]))
                # d=entrada[1]
                # if y!=d:
                        # incorrectos=incorrectos+1
                        # #print("ACTUALIZANDO PESOS")
                        # for i in range(len(pesos)):
                                # pesos[i]=pesos[i] + (tasa_aprendizaje * entrada[0][i] * (entrada[1] - y))
                                # #print("w"+str(i) + "=" + str(pesos[i]))
                        # #print("\n")


        # pincorrectos=incorrectos*1.0/len(entradas)
        # print(str(pincorrectos*100)+"% probability of error")
        # iter=iter+1

    # print("\nPesos finales:\n")

    # for i in range(len(pesos)):
        # print("w"+str(i) + "=" + str(pesos[i]))

    # entradas = []
    # for example in testdata:
        # criteria = [-1]
        # mood = 0
        # for word in example[0]:
            # if word in dictionary:
                # pass
            # else:
                # mood = mood + 1
            # answer = int(example[len(example)-1])
        # criteria.append(mood)
        # criteria.append(len(example[0]))
        # entradas.append(criteria)

    # for entrada in entradas:
        # y=funcion_activacion(funcion_nucleo(pesos,entrada))
        # if y == 0:
            # print("Sad phrase.")
        # else:
            # print("Not a sad phrase.")
        # print("\n")