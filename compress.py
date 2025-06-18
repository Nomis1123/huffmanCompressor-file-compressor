"""
Assignment 2 starter code
CSC148, Winter 2023

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2022 Bogdan Simion, Dan Zingaro
"""
from __future__ import annotations

import time

from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    return_dict = {}

    for item in text:
        return_dict[item] = return_dict.get(item, 0) + 1

    return return_dict


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """

    nodes = sorted(freq_dict.items(), key=lambda x: x[1])

    if len(nodes) == 1:
        valid_byte_nots = (nodes[0][0] + 1) % 256
        first_tree = HuffmanTree(nodes[0][0])
        second_tree = HuffmanTree(valid_byte_nots)
        parent_tree = HuffmanTree(None, first_tree, second_tree)
        return parent_tree

    while len(nodes) > 1:
        parent_tree = HuffmanTree()

        if isinstance(nodes[0][0], HuffmanTree):
            parent_tree.left = nodes[0][0]

        else:
            parent_tree.left = HuffmanTree(nodes[0][0])

        if isinstance(nodes[1][0], HuffmanTree):
            parent_tree.right = nodes[1][0]

        else:
            parent_tree.right = HuffmanTree(nodes[1][0])

        x = nodes[0][1]
        y = nodes[1][1]

        nodes = nodes[2:]

        nodes.append((parent_tree, x + y))

        nodes = sorted(nodes, key=lambda x: x[1])

    return nodes[0][0]


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """

    return __tree_codes(tree, '')


def __tree_codes(tree: HuffmanTree, code: str = '') -> dict[int, str]:
    """
    This function will return a dict with the code of the tree as the key
    and the symbol of the code as the int
    """
    left = tree.left
    right = tree.right

    if not left and not right:
        return {tree.symbol: code}

    return_dict = {}

    if left:
        return_dict.update(__tree_codes(left, code + '0'))

    if right:
        return_dict.update(__tree_codes(right, code + '1'))

    return return_dict


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """

    __num_tree(tree, 0)


def __num_tree(tree: HuffmanTree, num: int = 0) -> int:
    """
    This function will return the number of leaves within a given tree
    """
    left = tree.left
    right = tree.right

    if left:
        if not __is_leaf(left):
            num = __num_tree(left, num)

    if right:
        if not __is_leaf(right):
            num = __num_tree(right, num)

    if tree.number is None:
        tree.number = num
        num += 1
    return num


def __is_leaf(tree: HuffmanTree) -> bool:
    """
    This function checks if a tree node is a leaf
    """
    if not tree.left and not tree.right:
        return True
    else:
        return False


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """

    return_dict = get_codes(tree)
    x = 0
    y = 0
    for item in return_dict.items():
        x += len(item[1]) * freq_dict[item[0]]
        y += freq_dict[item[0]]

    return x / y


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """

    return_list = []
    bits_str = ""

    for item in text:
        bits_str += codes[item]

        while len(bits_str) >= 8:
            return_list.append(bits_to_byte(bits_str[:8]))
            bits_str = bits_str[8:]

    if not bits_str == '':
        return_list.append(bits_to_byte(bits_str))

    return bytes(return_list)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """

    return bytes(__tree_bytes(tree))


def __tree_bytes(tree: HuffmanTree) -> list:
    """
    This creates a big list of all the bytes within a given tree
    """
    left = tree.left
    right = tree.right
    return_list = []

    if left:
        if not __is_leaf(left):
            lst = __tree_bytes(left)
            for item in lst:
                return_list.append(item)

    if right:
        if not __is_leaf(right):
            lst = __tree_bytes(right)
            for item in lst:
                return_list.append(item)

    if tree.number >= 0:
        if left:
            if not __is_leaf(left):
                return_list.append(1)
                return_list.append(left.number)
            else:
                return_list.append(0)
                return_list.append(left.symbol)

        if right:
            if not __is_leaf(right):
                return_list.append(1)
                return_list.append(right.number)
            else:
                return_list.append(0)
                return_list.append(right.symbol)

    return return_list



def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """

    root = HuffmanTree()
    rn = node_lst[root_index]

    if rn.l_type == 1:
        root.left = generate_tree_general(node_lst, rn.l_data)
        root.left.number = rn.l_data

    else:
        root.left = HuffmanTree(rn.l_data)

    if rn.r_type == 1:
        root.right = generate_tree_general(node_lst, rn.r_data)
        root.right.number = rn.r_data
    else:
        root.right = HuffmanTree(rn.r_data)

    return root


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """

    left = None
    right = None

    for node in node_lst:

        root = HuffmanTree()
        if node.l_type == 0:
            root.left = HuffmanTree(node.l_data)

        else:
            if left:
                root.left = left
                root.left.number = node.l_data
                left = None

        if node.r_type == 0:
            root.right = HuffmanTree(node.r_data)

        else:
            if right:
                root.right = right
                root.right.number = node.r_data
                right = None

        if left and right:
            left = HuffmanTree(None, left, right)

        elif not left:
            left = root

        else:
            right = root

    if left:
        left.number = root_index
        return left

    else:
        right.number = root_index
        return right


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """

    bites = []
    c_byte = 0
    curr_byte = None
    c_bit = 0
    node = tree
    bits = ''
    while True:

        if curr_byte is None:
            curr_byte = text[c_byte]
            c_byte += 1
            c_bit = 0
            bits = byte_to_bits(curr_byte)

        if bits[c_bit] == '0':
            node = node.left
        else:
            node = node.right

        if __is_leaf(node):
            bites.append(node.symbol)
            node = tree

        c_bit += 1

        if c_bit == 8:
            curr_byte = None

        if len(bites) == size:
            break

    return bytes(bites)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """

    freaks = sorted(freq_dict.items(), key=lambda x: x[1])
    codes = get_codes(tree)

    while len(freaks) > 1:
        minn = freaks[0]
        maxx = freaks[-1]

        if len(codes[minn[0]]) < len(codes[maxx[0]]):
            minn_node = __find_node(tree, minn[0])
            maxx_node = __find_node(tree, maxx[0])

            if minn_node and maxx_node:
                temp1 = maxx_node.symbol
                temp2 = minn_node.symbol

                minn_node.symbol = temp1
                maxx_node.symbol = temp2
        freaks.pop(0)
        freaks.pop(-1)


def __find_node(tree: HuffmanTree, node: int) -> HuffmanTree:
    """
    This function finds a node within the given tree
    and then return that subtree
    """
    left = tree.left
    right = tree.right

    if left:
        if not __is_leaf(left):
            temp = __find_node(left, node)
            if temp:
                return temp
        else:
            if left.symbol == node:
                return left
    if right:
        if not __is_leaf(right):
            temp = __find_node(right, node)
            if temp:
                return temp
        else:
            if right.symbol == node:
                return right
    return None


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
