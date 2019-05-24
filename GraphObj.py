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

    def __init__(self, file_name):
        file_path = None

        if file_name is None:
            print("-E- Bad file name")
            exit()
        else:
            file_path = Path(file_name)

        if not file_path.is_file():
            print("-E- Bad path")
            exit()

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

        self.colors_array = np.full((self.V), -1).astype(int)

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
            x = self.resolve_last(domain_buffer[i], latest, i)

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

        print("-I- Time is ", time() - start)
        print("-I- Number of states is ", states)

        if states == consts.backtrack_itrations:
            print("backtrack terminated: itr reached upper bound")

        if i < 0:
            self.colors_array = prev
            return False
        else:
            return True

    def resolve_last(self, _current_domain, latest, i):
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
        print(self.consisntent_graph())
        print("Number of colors: ", len(set(colors)))
        print("Number of vertices = ", self.V)
        print("Number of edges = ", len(self.E))
        if print_colors:
            print("Coloring:", self.colors_array, end="")
        if print_edges:
            for e in self.E:
                print(e)

    def consisntent_graph(self):
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
                x = self.getArcConsistency(domain_buffer[i], i)

            if x is None:
                i = i - 1
                for k in range(i + 1, n):
                    domain_buffer[k] = deepcopy(dbuff_last[k])
            else:
                self.colors_array[i] = x
                i = i + 1

        if i < 0:
            self.colors_array = deepcopy(prev)
            print("total time: ", time() - start)
            print("states: ", states)
            return False
        else:
            print("total time: ", time() - start)
            print("states: ", states)
            return True

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

    def feasiblity_LS(self, itr=10, freedom=5, print_status=False):

        # apply greedy coloring method
        self.greedy_color(freedom)
        colors = list(set(self.colors_array))

        print("-I- Start checking form ", len(colors), " colors")

        # check initial consistency
        if not self.consisntent_graph():
            print("-I- Failed to color consistently")
            return 0

        states = 0
        start = time()
        consistent = True
        backup_array = deepcopy(self.colors_array)

        while consistent:
            if print_status:
                print("Attempting to reduce colors to ", len(colors), "...", end="")
            # count visitations in every v
            visits = np.zeros(shape=self.V).astype(int)

            # reduce number of colors by 1
            removed_color = choice(colors)
            colors.remove(removed_color)

            # reset color of vertices with the deleted color
            for i in range(0, self.V):
                if self.colors_array[i] == removed_color:
                    # print("something")
                    self.colors_array[i] = self.min_conflicts(colors, i)

            while True:
                # let v_i be a randomly selected conflicted vertex
                v_i = self.random_conflicted_vertex()
                states += 1

                # if there are no conflicts, continue
                if v_i == None:
                    if print_status:
                        print("Success")
                    backup_array = deepcopy(self.colors_array)
                    break

                # if v_i has been causing conlcits more than itr, we stop
                elif visits[v_i] > itr:
                    if print_status:
                        print("Failed")
                    consistent = False
                    break

                # update visits(Vi)
                # C(Vi) <- k s.t Vi has minimum conflicts with its neighbours
                visits[v_i] += 1
                self.colors_array[v_i] = self.min_conflicts(colors, v_i)

        # restore coloring to last feasible solution
        self.colors_array = deepcopy(backup_array)

        # finish
        print("Total time: ", time() - start)
        print("Total states: ", states)
        return True

    def num_of_colors(self):
        return len(set(self.colors_array))

    def max_color(self):
        return np.amax(self.colors_array)

    def greedy_color(self, freedom=1):
        # assign min color
        n = self.V
        for i in range(0, n):
            for k in range(0, n):

                if self.checkConsistent(i, k):
                    self.colors_array[i] = k
                    break

        # change each color according to freedom
        if (freedom == 1):
            return

        for i in range(0, n):
            color = self.colors_array[i]
            next_color = color + 1
            self.colors_array[i] = randint(color * freedom, next_color * freedom - 1)

    # Returns the color that minimizes conflicts for vertex i
    def min_conflicts(self, colors, i):

        # there are |V| vertices, so the max number of conflicts is |V|, therefore |V|+1 is greater than any number of possible conflicts
        min = self.V + 1
        best_color = -1

        # eval conflicts for Vi for each color, and return the color with the least conflicts
        for c in colors:
            conflicts = self.num_of_conflicts(i, c)
            if (conflicts < min):
                min = conflicts
                best_color = c

        return best_color

    def random_conflicted_vertex(self):
        conflicted_vertices = []

        # append all conflicted v to array
        for i in range(0, self.V):
            if (not self.checkConsistent(i)):
                conflicted_vertices.append(i)

        # if no conflictes, return -1
        if (len(conflicted_vertices) == 0):
            return None

        # output a random element from list of conflicted vertices
        return choice(conflicted_vertices)

    def num_of_conflicts(self, v, c):
        conflicts = 0
        for i in range(0, self.V):
            if (i == v):
                continue
            if self.checkEdge(v, i) and self.colors_array[i] == c:
                conflicts = conflicts + 1

        return conflicts

    def goal_target_Max_Ci(self, freedom_mod=5, print_progress=True):

        start = time()
        states = 0
        # apply some coloring
        self.greedy_color(freedom_mod)

        # back up coloring
        # backup_array = deepcopy(self.colors_array)

        # let C be a bucket list of all colors over V
        C = self.count_colors()
        if C is None:
            print("goal_target_Max_Ci exiting")
            return False

        # Init heap
        Heap = []
        self.queue_colors(Heap, C)

        # reduce colors by increasing sigma (Ci^2) for i=1..n
        while (Heap):

            if (print_progress):
                print("Attempting to decrease K to ", len(set(self.colors_array)) - 1, "...")

            # let c_min be the color we wish to remove
            c_min = self.pop(Heap)

            # for each vertex with color = c_min, change it
            color_changed = False
            for i in range(0, self.V):
                if self.colors_array[i] == c_min:
                    states += 1
                    x = self.select_value_goalTarget(i, C)
                    # if no such coloring exists, finish
                    if (x != None):
                        color_changed = True
                        self.colors_array[i] = x
                        if (not self.consisntent_graph()):
                            print("GRAPH NOT CONSISTENT")
                            return False
                        C[x] += 1
                        C[c_min] -= 1

            # if we successfully changed the graph, update the heap
            if (color_changed):
                Heap = []
                self.queue_colors(Heap, C)

        print("Total time: ", time() - start)
        print("Total states: ", states)

        return True

    # performs bucket counting of elements in colors_array, and returns it
    def count_colors(self):
        n = self.max_color()
        C = np.zeros(shape=(n + 1)).astype(int)
        for color in self.colors_array:
            if (color > n):
                print("count_colors error: color is greater than max range ", n)
                return None
            C[color] += 1
        return C

    # function initiate all colors (min first) into min heap H, beside the most frequent color
    def queue_colors(self, Heap, Colors):
        C_copy = deepcopy(Colors)
        c_max = np.argmax(C_copy).astype(int)
        for i in range(0, len(C_copy)):
            c_min = self.least_frequent_color(C_copy)
            # no more elements beside max (min=max)
            if (c_min == c_max):
                break

            elif (c_min == -1):
                print("queue colors: c_min not initiated")
                return

            # remove entry from C, and save it on H
            key = C_copy[c_min]
            C_copy[c_min] = 0
            self.push(Heap, key, c_min)

        del C_copy

    # given vertex i and a count of each color class, attempts to find a more frequent color for vertex i
    def select_value_goalTarget(self, i, colors_count):
        C = deepcopy(colors_count)
        this_color = self.colors_array[i]

        for j in range(0, len(C)):
            # get most frequenst color
            c_max = np.argmax(C).astype(int)
            c_max = int(c_max)

            # no more available colors
            if (c_max == this_color or c_max <= 0):
                return None

            # found consistent color
            if (self.checkConsistent(i, c_max)):
                return c_max

            # most frequent color inconsistent, remove it from temp list
            else:
                C[c_max] = 0

        # if all other colors are inconsistent, return None
        return None

    def least_frequent_color(self, C):
        min = self.V + 1
        min_color = -1
        for i in range(0, len(C)):
            if (C[i] < min and C[i] > 0):
                min = C[i]
                min_color = i

        return min_color

    def Target_LS(self, print_status=True):

        heap = []
        K = 0
        start = time()
        states = 0
        while (not self.fully_coloured()):
            states += 1
            if (print_status):
                print("Searching for ", K, "-th set")
            # insert all uncolored vertices to heap
            for i in range(0, self.V):
                if (self.uncolored(i)):
                    degree_i = self.neighbour_count(i)
                    self.push(heap, degree_i, i)

            # let S be a max independent set of uncolored vertices
            S = self.max_independent_set(heap)

            # Color S in K
            for s in S:
                self.colors_array[s] = K

            # update current color
            K = K + 1

        # finish
        print("Total time: ", time() - start)
        print("Total states: ", states)
        return True

    def uncolored(self, i):
        if (self.colors_array[i] == -1):
            return True
        return False

    # returns true when every vertex has a color
    def fully_coloured(self):
        for i in range(0, self.V):
            if (self.colors_array[i] == -1):
                return False
        return True

    def neighbour_count(self, i):
        res = 0
        for j in range(0, self.V):
            if (j == i):
                continue
            if (self.checkEdge(i, j)):
                res = res + 1

        return res

    def max_independent_set(self, heap):

        if (not heap):
            print("max set cover: heap emtpy")
            return None

        set = []
        set_neighbours = []

        # insert first vertex into set
        X_i = self.pop(heap)
        set.append(X_i)

        # insert all neighbours of X_1 to neighbourhood
        N = self.get_neighbours(X_i)
        set_neighbours = set + N

        # as long as heap is not empty, add independent vertices with lowest degree
        while (heap):
            # get next vertex with lowest degree
            X_i = self.pop(heap)

            # if X_i is indepent to current set, add X_i to set
            if (X_i not in set_neighbours):
                set.append(X_i)
                N = self.get_neighbours(X_i)

                # Add new neighbours to neighbourhood
                for i in range(0, N.__len__()):
                    if (N[i] not in set_neighbours):
                        set_neighbours.append(N[i])

        return set

    # returns all neighbors of vertex i
    def get_neighbours(self, i):
        neighbours = []

        # for vertex i, add all neighbours to array
        for j in range(0, self.V):
            if (i == j):
                continue
            if (self.checkEdge(i, j)):
                neighbours.append(j)
        # return neighbours
        return neighbours

    def hybrid(self, K, itr=5000, print_status=True):
        self.random_coloring(K)

        bad_vertices = self.get_conflicted_vertices()
        start = time()
        tries = 0
        while len(bad_vertices) > 0 and tries < itr:
            if (print_status and tries % 20 == 0):
                print("bad V: ", len(bad_vertices))
                print("K = ", len(set(self.colors_array)))
            tries += 1
            v = choice(bad_vertices)

            # choose new color for v
            new_color = self.select_value_hybrid(v)
            self.colors_array[v] = new_color

            bad_vertices = self.get_conflicted_vertices()

        print("Total time: ", time() - start)
        print("Total states: ", tries)

        if (tries < itr):
            return True
        else:
            return False

    def random_coloring(self, K):
        for i in range(0,self.V):
            self.colors_array[i] = randint(0,K-1)

    def get_conflicted_vertices(self):
        b = []
        for e in self.E:
            if (self.is_badEdge(e[0], e[1])):
                b.append(e[0])
                b.append(e[1])
        return b

    def select_value_hybrid(self, v):
        min = math.inf
        min_color = None
        C = self.count_colors()
        old_color = self.colors_array[v]
        C[old_color] -= 1
        for i in range(0, len(C)):
            if (C[i] == 0):
                continue
            self.colors_array[v] = i
            C[i] += 1
            B = self.count_bad_edges()
            fitness = self.calc_fitness(C, B)
            if fitness < min:
                min = fitness
                min_color = i
            C[i] -= 1

        return min_color

    def is_badEdge(self, i, j):
        if (i >= self.V):
            print("i out of range! ", i)
            return False
        if (j >= self.V):
            print("j out of range! ", j)
            return False

        if (self.colors_array[i] == self.colors_array[j]):
            return True
        else:
            return False

    # calc fitness according to:
    # SIGMA [2*Bi*Ci - SIGMA [Ci^2] ]
    def calc_fitness(self, C, B):
        # C: color counts
        # B: bad edges counts
        if (len(B) != len(C)):
            print("calc_fitness: B,C size mismatch!")
            return -1

        # calc fitness
        fitness = 0
        for i in range(0, len(B)):
            fitness += (2 * B[i] * C[i]) - C[i] ** 2

        return fitness

    def count_bad_edges(self):
        n = self.max_color()
        B = np.zeros(shape=(n + 1)).astype(int)
        for e in self.E:
            u = e[1]
            v = e[0]
            if (self.is_badEdge(u, v)):
                color = self.colors_array[u]
                B[color] += 1
        return B
