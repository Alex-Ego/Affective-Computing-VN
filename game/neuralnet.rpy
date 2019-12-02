################################################################################
## Neural Network Initialization
################################################################################

init offset = -3

init python:
    from __future__ import absolute_import, division, print_function, unicode_literals

    import tensorflow as tf

    import tensorflow_datasets as tfds
    import os
    def openfile(txtfile):
        lines = renpy.file(txtfile).readlines()
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
            if line != '':
                line = ''.join(line)
                line = str(line)
                cleanlines.append(line)
        cleanlines.pop(0)
        
        #Making the info useful
        cleandata = []
        for lines in cleanlines:
            lines = lines.split(", ")
            lines[0] = lines[0].split(" ")
            cleandata.append(lines)
        return cleandata
    
    def funcion_nucleo(pesos, entrada):
        arrayentradas = []
        for i in entrada:
                arrayentradas.append(entrada[i])
        return list(zip(pesos, arrayentradas))

    def funcion_activacion(resultado_nucleo):
        fx = sum(x*y for x, y in resultado_nucleo)
        if fx >= 0:
                return 1
        else:
                return 0
                
    max_iter=50
    tasa_aprendizaje=0.1
    pesos=[1,0,0,0,0,0,0]
    entradas=[[[-1,1,0,1,0,0,0], 1], [[-1,1,0,1,1,0,0], 1], [[-1,1,0,1,0,1,0], 1] 
    , [[-1,1,1,0,0,1,1], 1], [[-1,1,1,1,1,0,0], 1], [[-1,1,0,0,0,1,1], 1], [[-1,1,0,0,0,1,0], 0] 
    , [[-1,0,1,1,1,0,1], 1], [[-1,0,1,1,0,1,1], 0], [[-1,0,0,0,1,1,0], 0], [[-1,0,1,0,1,0,1], 0], 
    [[-1,0,0,0,1,0,1], 0], [[-1,0,1,1,0,1,1], 0], [[-1,0,1,1,1,0,0], 0]]

    iter=0
    pincorrectos=1.0

    while(pincorrectos>0.2 and iter<max_iter):

        incorrectos=0
        
        print("\nITERACION " + str(iter) + "\n")

        for entrada in entradas:
                print("Procesando entrada: ")
                print(entrada[0])
                print()
                y=funcion_activacion(funcion_nucleo(pesos,entrada[0]))
                d=entrada[1]
                if y!=d:
                        incorrectos=incorrectos+1
                        print("ACTUALIZANDO PESOS")
                        for i in range(len(pesos)):
                                pesos[i]=pesos[i] + (tasa_aprendizaje * entrada[0][i] * (entrada[1] - y))
                                print("w"+str(i) + "=" + str(pesos[i]))
                        print("\n")


        pincorrectos=incorrectos*1.0/len(entradas)
        print(str(pincorrectos*100)+"% de entradas procesadas incorrectamente")
        iter=iter+1

    print("\nPesos finales:\n")

    for i in range(len(pesos)):
        print("w"+str(i) + "=" + str(pesos[i]))
    
    print(clean_data((openfile("nndata/trainingdata.txt"))))