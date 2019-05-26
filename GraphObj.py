from pathlib import Path
import re
import numpy as np
from copy import deepcopy
import consts
from random import randint, choice
from time import time
from heapq import heappush
from heapq import heappop
import math


class GraphObj:

    def __init__(self, file_path):
        # file_path = None
        #
        # if file_name is None:
        #     print("-E- Bad file name")
        #     exit()
        # else:
        #     file_path = Path(file_name)

        # if not file_path.is_file():
        #     print("-E- Bad path")
        #     exit()

        fh = open(file_path, "rt")

        self.E = []
        local_E = 0

        lines = fh.readlines()
        for line in lines:
            if line[0] == 'p':
                [self.V, local_E] = self.getVals(line)
                self.G = np.zeros(shape=(self.V, self.V)).astype(bool)
            if line[0] == 'e':
                [i, j] = self.getVals(line)

                self.G[i - 1][j - 1] = 1

                self.E.append((i - 1, j - 1))

        self.colors_array = np.full(self.V, -1).astype(int)

        if len(self.E) != local_E:
            print("-E- Bad building - number of edges doesnt match to actual data")

    def getVals(self, line):
        if line is None:
            return [0, 0]

        m = re.search('\s(\d+)\s(\d+)', line)
        V = int(m.group(1))
        E = int(m.group(2))
        return [V, E]

    def backtrack(self, colors, print_value):
        states = 0
        prev = self.colors_array
        start = time()
        n = self.V
        i = 0

        domain = np.ones(shape=(self.V, colors)).astype(int)
        domain_buffer = deepcopy(domain)
        latest = np.full(self.V, -1).astype(int)

        backjumps = np.zeros(shape=self.V).astype(int)

        while (i >= 0) and (i < n) and (states < consts.backtrack_itrations):
            states += 1

            if print_value:
                print("-I- Index number  ", i)
            x = self.getLast(domain_buffer[i], latest, i)

            if x is None:
                i = latest[i]
                backjumps[i] += 1
                if backjumps[i] > 2 * colors:
                    i = -1

            else:
                self.colors_array[i] = x
                i = i + 1
                if i < n:
                    domain_buffer[i] = deepcopy(domain[i])
                    latest[i] = -1

        print("Time is ", time() - start)
        print("Number of states is ", states)

        if states == consts.backtrack_itrations:
            print("-W- Reached to ", consts.backtrack_itrations, " iterations")

        if i < 0:
            self.colors_array = prev
            return False
        else:
            return True

    def getLast(self, _current_domain, latest, i):
        current_domain = deepcopy(_current_domain)
        while not self.isEmpty(current_domain):
            a = self.getRandom(current_domain)
            current_domain[a] = 0
            consistent = True
            k = 0

            while k < i and consistent:
                if k > latest[i]:
                    latest[i] = k

                if self.checkConsistent(i, a):
                    k = k + 1
                else:
                    consistent = False

            if consistent:
                return a

        return None

    def isEmpty(self, _domain):
        if 1 in _domain:
            return False
        return True

    def getRandom(self, domain):
        indices = []
        n = domain.__len__()
        for i in range(0, n):
            if domain[i] == 1:
                indices.append(i)

        if indices.__len__() == 0:
            return -1

        x = randint(0, indices.__len__() - 1)
        return indices[x]

    def resetArray(self):
        for i in range(0, self.V):
            self.colors_array[i] = -1

    def checkConsistent(self, i, k=-1):
        if k == -1:
            k = self.colors_array[i]

        for j in range(0, self.V):
            if i == j:
                continue

            if self.checkEdge(i, j) and self.colors_array[j] == k:
                return False
        return True

    def checkEdge(self, i, j):
        if i > self.V or j > self.V:
            print("-E- Out of range (ce)")
            return None

        if self.G[i][j] == 1 or self.G[j][i] == 1:
            return True
        return False

    def printStats(self, print_edges=False, print_colors=False):
        colors = self.colors_array
        print("Consistent coloring: ", end="")
        print(self.checkConsisntentGraph())
        print("Number of colors: ", len(set(colors)))
        print("Number of vertices = ", self.V)
        print("Number of edges = ", len(self.E))
        if print_colors:
            print("Coloring:", self.colors_array, end="")
        if print_edges:
            for e in self.E:
                print(e)

    def checkConsisntentGraph(self):
        for i in range(0, self.V):
            for j in range(0, self.V):
                if i == j:
                    continue
                if self.checkEdge(i, j) and self.colors_array[i] == self.colors_array[j]:
                    return False
        return True

    def forwardcheck(self, colors, print_value):
        states = 0
        start = time()
        prev = deepcopy(self.colors_array)
        n = self.V
        i = 0

        domain = np.ones(shape=(self.V, colors)).astype(int)
        domain_buffer = deepcopy(domain)

        while (i < n) and (i >= 0):
            states += 1
            if print_value:
                print("-I- Index number  ", i)

            dbuff_last = deepcopy(domain_buffer)
            if i < n - 1:
                x = self.getArcConsistency(domain_buffer, i)
            else:
                x = self.selectLastValue(domain_buffer[i], i)

            if x is None:
                i = i - 1
                for k in range(i + 1, n):
                    domain_buffer[k] = deepcopy(dbuff_last[k])
            else:
                self.colors_array[i] = x
                i = i + 1

        if i < 0:
            self.colors_array = deepcopy(prev)
            print("Time is ", time() - start)
            print("Number of states is ", states)
            return False
        else:
            print("total time: ", time() - start)
            print("states: ", states)
            return True

    def selectLastValue(self, _domain, i):
        if self.isEmpty(_domain):
            # no available color for last vertex
            return None
        else:
            for c in range(0, len(_domain)):
                # color c is not in Domain of i
                if _domain[c] == 0:
                    continue
                # color c is consistent with last vertex; return it
                elif self.checkConsistent(i, c):
                    return c
            return None

    def multiple(self, colors, vertices):
        if len(vertices) != len(colors):
            print("-E- Vertices number and colors number are different")
            return False

        consistency = True

        backup_array = []
        for v in vertices:
            c = self.colors_array[v]
            backup_array.append(c)

        for i in range(0, len(colors)):
            v = vertices[i]
            self.colors_array[v] = colors[i]

        for v in vertices:
            if not consistency:
                break
            consistency = self.checkConsistent(v)

        for i in range(0, len(colors)):
            v = vertices[i]
            self.colors_array[v] = backup_array[i]

        return consistency

    def getArcConsistency(self, _current_domain, i):
        n = self.V
        domain_backup = deepcopy(_current_domain)

        while not self.isEmpty(_current_domain[i]):
            a = self.getRandom(_current_domain[i])
            _current_domain[i][a] = 0

            while True:
                removed_val = False

                for j in range(i + 1, n):
                    arc_consistent = False

                    for k in range(i + 1, n):
                        if arc_consistent:
                            break

                        for b in range(0, _current_domain[j].__len__()):
                            if arc_consistent:
                                break

                            if _current_domain[j][b] == 1:
                                for c in range(0, _current_domain[k].__len__()):

                                    if _current_domain[k][c] == 1:
                                        vertices = [i, j, k]
                                        colors = [a, b, c]
                                        arc_consistent = self.multiple(colors, vertices)

                                        if arc_consistent:
                                            break

                                    if (c == _current_domain[k].__len__() - 1) and (not arc_consistent):
                                        removed_val = True
                                        _current_domain[j][b] = 0

                if not removed_val:
                    break

            empty_domain = False
            for j in range(i + 1, n):
                if empty_domain:
                    break
                if self.isEmpty(_current_domain[j]):
                    empty_domain = True

            if empty_domain:
                for j in range(i + 1, n):
                    _current_domain[j] = deepcopy(domain_backup[j])
            else:
                return a

        return None

    def pop(self, heap):
        node = heappop(heap)
        return node[1]

    def push(self, heap, key, value):
        heappush(heap, (key, value))

    def feasiblity_LS(self, itr=10, freedom=5, print_val=False):

        # apply greedy coloring method
        self.setGreedyColor(freedom)
        colors = list(set(self.colors_array))

        print("-I- Start checking form ", len(colors), " colors")

        # check initial consistency
        if not self.checkConsisntentGraph():
            print("-I- Failed to color consistently")
            return 0

        states = 0
        consistent = True
        backup_array = deepcopy(self.colors_array)
        start = time()

        while consistent:
            if print_val:
                print("-I- Reduce colors to ", len(colors))
            # count visitations in every v
            visits_arr = np.zeros(shape=self.V).astype(int)

            # reduce number of colors by 1
            removed_color = choice(colors)
            colors.remove(removed_color)

            # reset color of vertices with the deleted color
            for i in range(0, self.V):
                if self.colors_array[i] == removed_color:
                    # print("something")
                    self.colors_array[i] = self.getMinConflictsColor(colors, i)

            while True:
                # let v_i be a randomly selected conflicted vertex
                rand_vertex = self.getRandomConflictedVertex()
                states += 1

                # if there are no conflicts, continue
                if not rand_vertex:
                    if print_val:
                        print("-I- Success")
                    backup_array = deepcopy(self.colors_array)
                    break

                # if v_i has been causing conflicts more than itr, we stop
                elif visits_arr[rand_vertex] > itr:
                    if print_val:
                        print("-I- Failed")
                    consistent = False
                    break

                # update visits(Vi)
                # C(Vi) <- k s.t Vi has minimum conflicts with its neighbours
                visits_arr[rand_vertex] += 1
                self.colors_array[rand_vertex] = self.getMinConflictsColor(colors, rand_vertex)

        # restore coloring to last feasible solution
        self.colors_array = deepcopy(backup_array)

        # finish
        print("-I- Summary:")
        print("Time: ", time() - start)
        print("States: ", states)
        return True

    def getColorsNumber(self):
        return len(set(self.colors_array))

    def getMaxColor(self):
        return np.amax(self.colors_array)

    def setGreedyColor(self, freedom=1):
        # assign min color
        n = self.V
        for i in range(0, n):
            for k in range(0, n):

                if self.checkConsistent(i, k):
                    self.colors_array[i] = k
                    break

        # change each color according to freedom
        if freedom == 1:
            return

        for i in range(0, n):
            color = self.colors_array[i]
            next_color = color + 1
            self.colors_array[i] = randint(color * freedom, next_color * freedom - 1)

    # Returns the color that minimizes conflicts for vertex i
    def getMinConflictsColor(self, _colors, i):

        # there are |V| vertices, so the max number of conflicts is |V|, therefore |V|+1 is greater than any number of possible conflicts
        minimum_value = self.V + 1
        best_color = -1

        # eval conflicts for Vi for each color, and return the color with the least conflicts
        for c in _colors:
            num_conflicts = self.getConflictsNumber(i, c)
            if num_conflicts < minimum_value:
                minimum_value = num_conflicts
                best_color = c

        return best_color

    def getRandomConflictedVertex(self):
        arr_conflicted_vertices = []

        # append all conflicted v to array
        for i in range(0, self.V):
            if not self.checkConsistent(i):
                arr_conflicted_vertices.append(i)

        # if no conflicts, return -1
        if len(arr_conflicted_vertices) == 0:
            return None

        # output a random element from list of conflicted vertices
        return choice(arr_conflicted_vertices)

    def getConflictsNumber(self, v, c):
        conflicts_counter = 0
        for i in range(0, self.V):
            if i == v:
                continue
            if self.checkEdge(v, i) and self.colors_array[i] == c:
                conflicts_counter = conflicts_counter + 1

        return conflicts_counter

    def solveTaretMax(self, freedom_mod=5, print_val=True):

        start = time()
        states = 0
        # apply some coloring
        self.setGreedyColor(freedom_mod)

        # back up coloring
        # backup_array = deepcopy(self.colors_array)

        # let color_count be a bucket list of all colors over V
        color_count = self.countColors()
        if color_count is None:
            return False

        # Init heap
        heap = []
        self.initHeap(heap, color_count)

        # reduce colors by increasing sigma (Ci^2) for i=1..n
        while heap:

            if print_val:
                print("-I- Decrease K to ", len(set(self.colors_array)) - 1)

            # let c_min be the color we wish to remove
            c_min = self.pop(heap)

            # for each vertex with color = c_min, change it
            color_changed = False
            for i in range(0, self.V):
                if self.colors_array[i] == c_min:
                    states += 1
                    x = self.selectValueToGoalTarget(i, color_count)
                    # if no such coloring exists, finish
                    if x:
                        color_changed = True
                        self.colors_array[i] = x
                        if not self.checkConsisntentGraph():
                            print("-E- Graph is not consistent")
                            return False
                        color_count[x] += 1
                        color_count[c_min] -= 1

            # if we successfully changed the graph, update the heap
            if color_changed:
                heap = []
                self.initHeap(heap, color_count)

        print("-I- Summary:")
        print("Time: ", time() - start)
        print("States: ", states)

        return True

    # performs bucket counting of elements in colors_array, and returns it
    def countColors(self):
        n = self.getMaxColor()
        C = np.zeros(shape=(n + 1)).astype(int)
        for color in self.colors_array:
            if color > n:
                print("-E- Color is greater than max range ", n)
                return None
            C[color] += 1
        return C

    # function initiate all colors (min first) into min heap H, beside the most frequent color
    def initHeap(self, _heap, _colors):
        array_backup = deepcopy(_colors)
        getMaxColor = np.argmax(array_backup).astype(int)
        for i in range(0, len(array_backup)):
            min_color = self.geLeastFrequentColor(array_backup)
            # no more elements beside max (min=max)
            if min_color == getMaxColor:
                break

            elif min_color == -1:
                print("-W- Variable \"min_color\" is not initiated")
                return

            # remove entry from C, and save it on H
            key = array_backup[min_color]
            array_backup[min_color] = 0
            self.push(_heap, key, min_color)

        del array_backup

    # given vertex i and a count of each color class, attempts to find a more frequent color for vertex i
    def selectValueToGoalTarget(self, i, colors_count):
        backup = deepcopy(colors_count)
        this_color = self.colors_array[i]

        for j in range(0, len(backup)):
            # get most frequenst color
            getMaxColor = np.argmax(backup).astype(int)
            getMaxColor = int(getMaxColor)

            # no more available colors
            if getMaxColor == this_color or getMaxColor <= 0:
                return None

            # found consistent color
            if self.checkConsistent(i, getMaxColor):
                return getMaxColor

            # most frequent color inconsistent, remove it from temp list
            else:
                backup[getMaxColor] = 0

        # if all other colors are inconsistent, return None
        return None

    def geLeastFrequentColor(self, _colors_arr):
        minimum_value = self.V + 1
        minimum_color = -1
        for i in range(0, len(_colors_arr)):
            if (_colors_arr[i] < minimum_value) and (_colors_arr[i] > 0):
                minimum_value = _colors_arr[i]
                minimum_color = i

        return minimum_color

    def Target_LS(self, print_val=True):

        heap = []
        K = 0
        start = time()
        states = 0
        while not self.setAllColoured():
            states += 1
            if print_val:
                print("-I- Check for the ", K, "# set")
            # insert all uncolored vertices to heap
            for i in range(0, self.V):
                if self.uncolored(i):
                    degree_i = self.countNeighbour(i)
                    self.push(heap, degree_i, i)

            # let S be a max independent set of uncolored vertices
            S = self.getMaxIndependentSet(heap)

            # Color S in K
            for s in S:
                self.colors_array[s] = K

            # update current color
            K = K + 1

        # finish
        print("-I- Summary:")
        print("Time: ", time() - start)
        print("States: ", states)
        return True

    def uncolored(self, i):
        if self.colors_array[i] == -1:
            return True
        return False

    # returns true when every vertex has a color
    def setAllColoured(self):
        for i in range(0, self.V):
            if self.colors_array[i] == -1:
                return False
        return True

    def countNeighbour(self, i):
        res = 0
        for j in range(0, self.V):
            if j == i:
                continue
            if self.checkEdge(i, j):
                res = res + 1

        return res

    def getMaxIndependentSet(self, heap):

        if not heap:
            print("-E- Heap is emtpy")
            return None

        set = []

        # insert first vertex into set
        top_element = self.pop(heap)
        set.append(top_element)

        # insert all neighbours of top_element to neighbourhood
        element_neighbours = self.getNeighbours(top_element)
        set_neighbours = set + element_neighbours

        # as long as heap is not empty, add independent vertices with lowest degree
        while heap:
            # get next vertex with lowest degree
            top_element = self.pop(heap)

            # if top_element is indepent to current set, add top_element to set
            if top_element not in set_neighbours:
                set.append(top_element)
                element_neighbours = self.getNeighbours(top_element)

                # Add new neighbours to neighbourhood
                for i in range(0, element_neighbours.__len__()):
                    if element_neighbours[i] not in set_neighbours:
                        set_neighbours.append(element_neighbours[i])

        return set

    # returns all neighbors of vertex i
    def getNeighbours(self, i):
        neighbours = []

        # for vertex i, add all neighbours to array
        for j in range(0, self.V):
            if i == j:
                continue
            if self.checkEdge(i, j):
                neighbours.append(j)
        # return neighbours
        return neighbours

    def hybrid(self, K, itr=5000, print_status=True):
        self.setRandomColors(K)

        bad_vertices = self.getConflictedVertices()
        start = time()
        attempts_counter = 0
        while len(bad_vertices) > 0 and attempts_counter < itr:
            if print_status and attempts_counter % 20 == 0:
                print("-I- Number of bad vertices: ", len(bad_vertices))
                print("-I- K value is ", len(set(self.colors_array)))
            attempts_counter += 1
            v = choice(bad_vertices)

            # choose new color for v
            new_color = self.getHybridValue(v)
            self.colors_array[v] = new_color

            bad_vertices = self.getConflictedVertices()

        print("-I- Summary:")
        print("Time: ", time() - start)
        print("States: ", attempts_counter)

        if attempts_counter < itr:
            return True

        return False

    def setRandomColors(self, K):
        for i in range(0, self.V):
            self.colors_array[i] = randint(0, K - 1)

    def getConflictedVertices(self):
        conflicted_array = []
        for e in self.E:
            if self.checkBadEdge(e[0], e[1]):
                conflicted_array.append(e[0])
                conflicted_array.append(e[1])
        return conflicted_array

    def getHybridValue(self, v):
        minimum = math.inf
        minimum_color = None
        num_colors = self.countColors()
        backup = self.colors_array[v]
        num_colors[backup] -= 1
        for i in range(0, len(num_colors)):
            if num_colors[i] == 0:
                continue
            self.colors_array[v] = i
            num_colors[i] += 1
            be_number = self.countBadEdgers()
            fitness = self.calcFitness(num_colors, be_number)
            if fitness < minimum:
                minimum = fitness
                minimum_color = i
            num_colors[i] -= 1

        return minimum_color

    def checkBadEdge(self, i, j):
        if j >= self.V:
            print("-E- Index j out of range! ", j)
            return False
        if i >= self.V:
            print("-E- Index i out of range! ", i)
            return False

        if self.colors_array[i] == self.colors_array[j]:
            return True
        else:
            return False

    # calc fitness according to:
    # SIGMA [2*Bi*Ci - SIGMA [Ci^2] ]
    def calcFitness(self, _colors, _bad_edges):
        # C: color counts
        # B: bad edges counts
        if len(_bad_edges) != len(_colors):
            print("-W- Color counter and bad edges counter not equal")
            return -1

        # calc fitness
        fitness = 0
        for i in range(0, len(_bad_edges)):
            fitness += (2 * _bad_edges[i] * _colors[i]) - _colors[i] ** 2

        return fitness

    def countBadEdgers(self):
        getMaxColor = self.getMaxColor()
        bad_edges_arr = np.zeros(shape=(getMaxColor + 1)).astype(int)
        for e in self.E:
            u = e[1]
            v = e[0]
            if self.checkBadEdge(u, v):
                color = self.colors_array[u]
                bad_edges_arr[color] += 1
        return bad_edges_arr
