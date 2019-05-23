from GraphObj import GraphObj
from time import time

def solve_backtrack(gObj, print):
    K = 0
    start = time()
    for k in range(4, 100):
        gObj.resetArray()
        if (gObj.backtrack(k, print)):
            K = k
            break

    total = time() - start
    return [K, total]

def Backtrack(file_name, type=1, print=True):
    graphObj = GraphObj(filename)
    graphObj.resetArray()
    colors = 0
    time = 0

    if type == 1:
        [colors, time] = solve_backtrack(graphObj, print)


    elif type == 2:
        [colors, time] = solve_backtrack_FC(graphObj, print)

    else:
        print("-E- Only types are 1 or 2")
        return

    printResults(graphObj, colors, time)

def printResults(graphObj, colors, time):
    print("\nGraph statistics: ")
    graphObj.printStats()
    print("\nChromatic color: ", colors)
    print("\nTime: ", time)


filename = "DSJC125.1.col"

# backtrack search
print("attempting backtrack with backjumping search...")
Backtrack(filename, 1, False)

# print("attempting backtrack with arc consistency search...")
# Backtrack(filename, type=2, print_status=False)
#
# # local search
# print("attempting feasibility approach local search...")
# local_search(filename, type=1, print_status=False, itr=100, visits=10, freedom=5)
#
# print("attempting goal target approach local search...")
# local_search(filename, type=2, print_status=False, itr=100, visits=10, freedom=5)
#
# print("attempting goal target (max independent set) approach local search...")
# local_search(filename, type=3, print_status=False, itr=100, visits=10, freedom=5)
#
# print("attempting hybrid approach (bad edges fitness) local search...")
# local_search(filename, type=4, print_status=False, itr=100, visits=10, freedom=5)
