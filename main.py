from GraphObj import GraphObj
from time import time
from copy import deepcopy
import os.path
EPOCHS = 5

def Backtrack(file_name, print_val):
    graphObj = GraphObj(filename)
    graphObj.resetArray()
    c_color = 0
    start = time()
    for k in range(4, 100):
        graphObj.resetArray()
        if graphObj.backtrack(k, print_val):
            c_color = k
            break

    total = time() - start

    printResults(graphObj, c_color, total)


def Forward_checking(filename, print_val=False):
    graphObj = GraphObj(filename)
    graphObj.resetArray()
    start = time()
    c_color = 0

    for k in range(4, 100):
        graphObj.resetArray()
        print("-I- Check for ", k, " colors")
        if graphObj.forwardcheck(k, print_val):
            c_color = k
            break

    total = time() - start
    printResults(graphObj, c_color, total)


def Feasibility(filename, itr=10, freedom=5, print_val=False):
    graphObj = GraphObj(filename)
    minimum = graphObj.V + 1
    start = time()
    arr_coloring = []
    for i in range(0, EPOCHS):
        graphObj.resetArray()
        graphObj.feasiblity_LS(itr, freedom, print_val)
        if graphObj.getColorsNumber() < minimum:
            minimum = graphObj.getColorsNumber()
            arr_coloring = deepcopy(graphObj.colors_array)

    total = time() - start
    graphObj.colors_array = deepcopy(arr_coloring)
    printResults(graphObj, minimum, total)


def Target(filename, freedom=5, print_val=False):
    graphObj = GraphObj(filename)
    colors_num = graphObj.V
    arr_coloring = deepcopy(graphObj.colors_array)
    start = time()

    for i in range(0, EPOCHS):
        graphObj.Target_LS(freedom, print_val)
        k = graphObj.getColorsNumber()
        if k < colors_num:
            colors_num = k
            arr_coloring = deepcopy(graphObj.colors_array)

    t = time() - start
    graphObj.colors_array = deepcopy(arr_coloring)
    del arr_coloring
    printResults(graphObj, colors_num, t)


def Hybrid(filename, itr=100, print_val=False):
    graphObj = GraphObj(filename)
    colors_num = 0
    start = time()

    for k in range(5, 100):
        graphObj.resetArray()
        if graphObj.hybrid(k, itr, print_val):
            colors_num = graphObj.getColorsNumber()
            break

    total = time() - start
    printResults(graphObj, colors_num, total)


def printResults(graphObj, colors, time):
    print("\nGraph statistics: ")
    graphObj.printStats()
    print("\nChromatic color: ", colors)
    print("\nTime: ", time)


path = os.getcwd() + "\\graphs"

col_files = os.listdir(path)
run_all = False
for filename in col_files:
    if not run_all:
        continueVal = input(
            "\nThe following file is \"" + filename + "\"\nPress A to RUN ALL, "
                                                      "R to RUN one-by-one , "
                                                      "S to SKIP this file "
                                                      "or E to EXIT"
                                                      "\n(by default will run all)\n")

    if not continueVal == 'r' or continueVal == 'R':
        if continueVal == 's' or continueVal == 'S':
            print("-I- Skip " + filename)
            continue
        if continueVal == 'e' or continueVal == 'E':
            print("-I- Exit")
            break
        if continueVal == 'a' or continueVal == 'A':
            run_all = True

    print("-I- Run \"" + filename + "\" file's graph\n")
    filename = os.getcwd() + "\\graphs\\" + filename
    # backtrack search vs forward checking
    print("-I- Start backtracking with backjumping")
    print("****************************************")
    Backtrack(filename, False)

    print("\n-I- Start forward checking with arc consistency")
    print("*************************************************")
    Forward_checking(filename, False)

    # local search
    print("\n-I- Local search part")
    print("-I- Feasibility")
    print("****************")
    Feasibility(filename, 10, 5, False)

    print("\n-I- Target function")
    print("*********************")
    Target(filename, 5, False)

    print("\n-I- Hybrid")
    print("***********")
    Hybrid(filename, 100, False)

    print("**********************************************************************************")

print("-I- Done")
