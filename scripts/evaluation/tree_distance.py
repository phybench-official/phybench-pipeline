#!/usr/bin/env python
# Original Authors: Tim Henderson and Steve Johnson
# Email: tim.tadh@gmail.com, steve@steveasleep.com
# For licensing see the LICENSE file in the top level directory.

# This is a modified version of zss package.

from __future__ import annotations

import collections
from collections.abc import Callable
from typing import Any

from numpy import ones, zeros


class Node:
    def __init__(self, label: str, children: list[Node] | None = None) -> None:
        self.label = label
        self.children = children or []

    @staticmethod
    def get_children(node: Node) -> list[Node]:
        return node.children

    @staticmethod
    def get_label(node: Node) -> str:
        return node.label

    def addkid(self, node: Node, before: bool = False) -> Node:
        if before:
            self.children.insert(0, node)
        else:
            self.children.append(node)
        return self

    def get(self, label: str) -> Node | None:
        if self.label == label:
            return self
        for c in self.children:
            result = c.get(label)
            if result is not None:
                return result
        return None


class AnnotatedTree:
    def __init__(self, root: Any, get_children: Callable[[Any], list[Any]]) -> None:
        self.get_children = get_children

        self.root = root
        self.nodes: list[Any] = []  # a post-order enumeration of the nodes in the tree
        self.ids: list[int] = []  # a matching list of ids
        self.lmds: list[int] = []  # left most descendents of each nodes
        self.keyroots: list[int] = []  # the keyroots in the original paper

        stack: list[tuple[Any, collections.deque[int]]] = []
        pstack: list[tuple[tuple[Any, int], collections.deque[int]]] = []
        stack.append((root, collections.deque()))
        j = 0
        while len(stack) > 0:
            n, anc = stack.pop()
            nid = j
            for c in self.get_children(n):
                anc_copy: collections.deque[int] = collections.deque(anc)
                anc_copy.appendleft(nid)
                stack.append((c, anc_copy))
            pstack.append(((n, nid), anc))
            j += 1
        lmds: dict[int, int] = {}
        keyroots: dict[int, int] = {}
        i = 0
        while len(pstack) > 0:
            (n, nid), anc = pstack.pop()
            self.nodes.append(n)
            self.ids.append(nid)
            if not self.get_children(n):
                lmd = i
                for a in anc:
                    if a not in lmds:
                        lmds[a] = i
                    else:
                        break
            else:
                try:
                    lmd = lmds[nid]
                except KeyError:
                    lmd = i  # Default fallback if nid not found
            self.lmds.append(lmd)
            keyroots[lmd] = i
            i += 1
        self.keyroots = sorted(keyroots.values())


def ext_distance(
    A: Any,
    B: Any,
    get_children: Callable[[Any], list[Any]],
    single_insert_cost: Callable[[Any], float],
    insert_cost: Callable[[Any], float],
    single_remove_cost: Callable[[Any], float],
    remove_cost: Callable[[Any], float],
    update_cost: Callable[[Any, Any], float],
) -> float:
    """Computes the extended tree edit distance between trees A and B with extended-zss algorithm
    Args:
        A(Node): Root node of tree 1
        B(Node): Root node of tree 2
        get_children(Func): the get_children method of tree
        single_insert_cost(Func): cost of inserting single node
        insert_cost(Func): cost of inserting a subtree
        update_cost(Func): cost of updating A to B


    Return:
        Distance(float): the tree editing distance
    """
    tree_A, tree_B = AnnotatedTree(A, get_children), AnnotatedTree(B, get_children)
    size_a = len(tree_A.nodes)
    size_b = len(tree_B.nodes)
    treedists = zeros((size_a, size_b), float)
    fd = 1000 * ones((size_a + 1, size_b + 1), float)

    def treedist(x: int, y: int) -> None:
        Al = tree_A.lmds
        Bl = tree_B.lmds
        An = tree_A.nodes
        Bn = tree_B.nodes

        m = size_a

        fd[Al[x]][Bl[y]] = 0
        for i in range(Al[x], x + 1):
            node = An[i]
            fd[i + 1][Bl[y]] = fd[Al[i]][Bl[y]] + remove_cost(node)

        for j in range(Bl[y], y + 1):
            node = Bn[j]

            fd[Al[x]][j + 1] = fd[Al[x]][Bl[j]] + insert_cost(node)

        for i in range(Al[x], x + 1):
            for j in range(Bl[y], y + 1):
                node1 = An[i]
                node2 = Bn[j]
                costs = [
                    fd[i][j + 1] + single_remove_cost(node1),
                    fd[i + 1][j] + single_insert_cost(node2),
                    fd[Al[i]][j + 1] + remove_cost(node1),
                    fd[i + 1][Bl[j]] + insert_cost(node2),
                ]
                m = min(costs)

                if Al[x] == Al[i] and Bl[y] == Bl[j]:
                    treedists[i][j] = min(m, fd[i][j] + update_cost(node1, node2))
                    fd[i + 1][j + 1] = treedists[i][j]
                else:
                    fd[i + 1][j + 1] = min(m, fd[Al[i]][Bl[j]] + treedists[i][j])

    for x in tree_A.keyroots:
        for y in tree_B.keyroots:
            treedist(x, y)

    return float(treedists[-1][-1])
