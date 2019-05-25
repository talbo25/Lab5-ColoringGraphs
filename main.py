from GraphObj import GraphObj
from time import time
from copy import deepcopy
import consts

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
    for i in range(0, consts.epochs):
        graphObj.resetArray()
        graphObj.feasiblity_LS(itr, freedom, print_val)
        if graphObj.getColorsNumber() < minimum:
            minimum = graphObj.getColorsNumber()
            arr_coloring = deepcopy(graphObj.colors_array)

    total = time() - start
    graphObj.colors_array = deepcopy(arr_coloring)
    printResults(graphObj, minimum, total)

#type 3 goal target (max independent set)
def Target(filename):
    graphObj = GraphObj(filename)
    graphObj.resetArray()
    start = time()
    graphObj.Target_LS()
    colors_num = graphObj.getColorsNumber()
    total = time() - start
    printResults(graphObj, colors_num, total)


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

#type 2 goal target (max Ci^2)
def solve_max_Ci(filename, freedom=5, print_val=False):
    graphObj = GraphObj(filename)
    colors_num = graphObj.V
    arr_coloring = deepcopy(graphObj.colors_array)
    start = time()

    for i in range(0, consts.epochs):
        graphObj.solveTaretMax(freedom, print_val)
        k = graphObj.getColorsNumber()
        if k < colors_num:
            colors_num = k
            arr_coloring = deepcopy(graphObj.colors_array)

    t = time() - start
    graphObj.colors_array = deepcopy(arr_coloring)
    del arr_coloring
    printResults(graphObj, colors_num, t)


def printResults(graphObj, colors, time):
    print("\nGraph statistics: ")
    graphObj.printStats()
    print("\nChromatic color: ", colors)
    print("\nTime: ", time)


filename = "DSJC125.1.col"

# # backtrack search vs forward checking
# print("-I- Start backtracking with backjumping")
# Backtrack(filename, False)

print("-I- Start forward checking with arc consistency")
Forward_checking(filename, False)

# # local search
# print("-I- Local search part")
# print("-I- Feasibility")
# Feasibility(filename, 10, 5, False)
#
# # goal target (max independent set)
# print("-I- Target function")
# Target(filename)
#
# print("-I- Hybrid")
# Hybrid(filename, 100, False)
#
# # goal target (max Ci^2)
# print("attempting goal target approach local search...")
# solve_max_Ci(filename, 5, False)
