from pathlib import Path
import re
from time import time
import numpy as np
from copy import deepcopy
import consts
from random import randint, choice
from time import time


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

            if self.checkEdge(i, j) and self.colors_array[i] == k:
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
