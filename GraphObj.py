import re
import numpy as np
from copy import deepcopy
from random import randint, choice
from time import time
from heapq import heappush
from heapq import heappop
import math
BACKTRACK_ITERATIONS = 50000
EPOCHS = 5

class GraphObj:

    def __init__(self, file_path):
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

        if not self.V:
            print("-E- Problem with init graph")
            exit(1)
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

        while (i >= 0) and (i < n) and (states < BACKTRACK_ITERATIONS):
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

        if states == BACKTRACK_ITERATIONS:
            print("-W- Reached to ", BACKTRACK_ITERATIONS, " iterations")

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
        print("Number of colors = ", len(set(colors)))
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
            return None
        else:
            for c in range(0, len(_domain)):
                if _domain[c] == 0:
                    continue
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

        self.setGreedyColor(freedom)
        colors = list(set(self.colors_array))

        print("-I- Start checking form ", len(colors), " colors")

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
            visits_arr = np.zeros(shape=self.V).astype(int)

            if not colors:
                print("-W- Empty list. start again")
                # consistent = False
                break

            removed_color = choice(colors)
            colors.remove(removed_color)

            for i in range(0, self.V):
                if self.colors_array[i] == removed_color:
                    self.colors_array[i] = self.getMinConflictsColor(colors, i)

            while True:
                rand_vertex = self.getRandomConflictedVertex()
                states += 1

                if not rand_vertex:
                    if print_val:
                        print("-I- Success")
                    backup_array = deepcopy(self.colors_array)
                    break

                elif visits_arr[rand_vertex] > itr:
                    if print_val:
                        print("-I- Failed")
                    consistent = False
                    break

                visits_arr[rand_vertex] += 1
                self.colors_array[rand_vertex] = self.getMinConflictsColor(colors, rand_vertex)

        self.colors_array = deepcopy(backup_array)

        print("-I- Summary:")
        print("Time: ", time() - start)
        print("States: ", states)
        return True

    def getColorsNumber(self):
        return len(set(self.colors_array))

    def getMaxColor(self):
        return np.amax(self.colors_array)

    def setGreedyColor(self, freedom=1):
        n = self.V
        for i in range(0, n):
            for k in range(0, n):

                if self.checkConsistent(i, k):
                    self.colors_array[i] = k
                    break

        if freedom == 1:
            return

        for i in range(0, n):
            color = self.colors_array[i]
            next_color = color + 1
            self.colors_array[i] = randint(color * freedom, next_color * freedom - 1)

    def getMinConflictsColor(self, _colors, i):
        minimum_value = self.V + 1
        best_color = -1

        for c in _colors:
            num_conflicts = self.getConflictsNumber(i, c)
            if num_conflicts < minimum_value:
                minimum_value = num_conflicts
                best_color = c

        return best_color

    def getRandomConflictedVertex(self):
        arr_conflicted_vertices = []

        for i in range(0, self.V):
            if not self.checkConsistent(i):
                arr_conflicted_vertices.append(i)

        if len(arr_conflicted_vertices) == 0:
            return None

        return choice(arr_conflicted_vertices)

    def getConflictsNumber(self, v, c):
        conflicts_counter = 0
        for i in range(0, self.V):
            if i == v:
                continue
            if self.checkEdge(v, i) and self.colors_array[i] == c:
                conflicts_counter = conflicts_counter + 1

        return conflicts_counter

    def Target_LS(self, freedom_mod=5, print_val=True):

        start = time()
        states = 0
        self.setGreedyColor(freedom_mod)

        color_count = self.countColors()
        if color_count is None:
            return False

        heap = []
        self.initHeap(heap, color_count)

        while heap:
            if print_val:
                print("-I- Decrease K to ", len(set(self.colors_array)) - 1)

            c_min = self.pop(heap)

            color_changed = False
            for i in range(0, self.V):
                if self.colors_array[i] == c_min:
                    states += 1
                    x = self.selectValueToGoalTarget(i, color_count)
                    if x:
                        color_changed = True
                        self.colors_array[i] = x
                        if not self.checkConsisntentGraph():
                            print("-E- Graph is not consistent")
                            return False
                        color_count[x] += 1
                        color_count[c_min] -= 1

            if color_changed:
                heap = []
                self.initHeap(heap, color_count)

        print("-I- Summary:")
        print("Time: ", time() - start)
        print("States: ", states)

        return True

    def countColors(self):
        n = self.getMaxColor()
        C = np.zeros(shape=(n + 1)).astype(int)
        for color in self.colors_array:
            if color > n:
                print("-E- Color is greater than max range ", n)
                return None
            C[color] += 1
        return C

    def initHeap(self, _heap, _colors):
        array_backup = deepcopy(_colors)
        getMaxColor = np.argmax(array_backup).astype(int)
        for i in range(0, len(array_backup)):
            min_color = self.geLeastFrequentColor(array_backup)
            if min_color == getMaxColor:
                break

            elif min_color == -1:
                print("-W- Variable \"min_color\" is not initiated")
                return

            key = array_backup[min_color]
            array_backup[min_color] = 0
            self.push(_heap, key, min_color)

        del array_backup

    def selectValueToGoalTarget(self, i, colors_count):
        backup = deepcopy(colors_count)
        this_color = self.colors_array[i]

        for j in range(0, len(backup)):
            getMaxColor = np.argmax(backup).astype(int)
            getMaxColor = int(getMaxColor)

            if getMaxColor == this_color or getMaxColor <= 0:
                return None

            if self.checkConsistent(i, getMaxColor):
                return getMaxColor
            else:
                backup[getMaxColor] = 0

        return None

    def geLeastFrequentColor(self, _colors_arr):
        minimum_value = self.V + 1
        minimum_color = -1
        for i in range(0, len(_colors_arr)):
            if (_colors_arr[i] < minimum_value) and (_colors_arr[i] > 0):
                minimum_value = _colors_arr[i]
                minimum_color = i

        return minimum_color

    def hybrid(self, N, itr=5000, print_status=True):
        self.setRandomColors(N)

        bad_vertices = self.getConflictedVertices()
        start = time()
        attempts_counter = 0
        while len(bad_vertices) > 0 and attempts_counter < itr:
            if print_status and attempts_counter % 20 == 0:
                print("-I- Number of bad vertices: ", len(bad_vertices))
                print("-I- N value is ", len(set(self.colors_array)))
            attempts_counter += 1
            v = choice(bad_vertices)

            new_color = self.getHybridValue(v)
            self.colors_array[v] = new_color

            bad_vertices = self.getConflictedVertices()

        print("-I- Summary:")
        print("Time: ", time() - start)
        print("States: ", attempts_counter)

        if attempts_counter < itr:
            return True

        return False

    def setRandomColors(self, N):
        for i in range(0, self.V):
            self.colors_array[i] = randint(0, N - 1)

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

    def calcFitness(self, _colors, _bad_edges):
        if len(_bad_edges) != len(_colors):
            print("-W- Color counter and bad edges counter not equal")
            return -1

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
