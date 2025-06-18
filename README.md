# Huffman Compressor - A File Compression Tool in Python

This project is a from-scratch implementation of the classic Huffman coding algorithm for lossless data compression, written entirely in Python. It is designed to demonstrate a deep understanding of core computer science concepts, including data structures (binary trees, priority queues), algorithmic design, bit-level data manipulation, file I/O, and robust, property-based testing.

## Features

*   **File Compression & Decompression:**
    *   Compresses any given file into a custom `.huf` format.
    *   Decompresses `.huf` files back to their original state, ensuring perfect data integrity.
*   **Classic Huffman Algorithm Implementation:**
    *   **Frequency Analysis:** Scans the input file to build a frequency dictionary of each byte (`build_frequency_dict`).
    *   **Huffman Tree Construction:** Builds an optimal prefix-code binary tree based on byte frequencies, treating the frequency map as a priority queue (`build_huffman_tree`).
    *   **Code Generation:** Traverses the Huffman tree to generate a unique binary code for each symbol, with more frequent symbols receiving shorter codes (`get_codes`).
*   **Efficient Bit-Level Packing:**
    *   The generated binary codes are packed tightly into bytes, minimizing wasted space. This goes beyond simply storing '0's and '1's as characters (`compress_bytes`).
*   **Tree Serialization:**
    *   The structure of the generated Huffman tree is serialized into a compact byte representation and stored in the header of the `.huf` file. This is crucial, as the exact tree is required for decompression (`tree_to_bytes`).
*   **Command-Line Interface:**
    *   Provides a simple interactive prompt to choose between compressing (`c`) or decompressing (`d`) a file.
*   **Tree Shape Optimization (Advanced):**
    *   Includes an `improve_tree` function that intelligently swaps symbols within the tree's leaf nodes—without changing the tree's structure—to better align symbol frequencies with code lengths, further optimizing the compression ratio.
*   **Robust Property-Based Testing:**
    *   The project includes a comprehensive test suite using `pytest` and the `hypothesis` library.
    *   Tests verify not just simple cases but the fundamental properties of the implementation, including a crucial "round-trip" test that ensures `decompress(compress(data)) == data` for any binary input.

## The `.huf` Custom File Format

The compressor generates files with a custom `.huf` extension. The file has a simple header followed by the data payload, allowing it to be self-contained and easily decompressed.

1.  **Number of Tree Nodes (1 byte):** The number of internal nodes in the serialized Huffman Tree.
2.  **Serialized Tree Data (`N * 4` bytes):** A post-order traversal of the tree's internal nodes, where each node is represented by 4 bytes. This allows the exact tree to be reconstructed during decompression.
3.  **Original File Size (4 bytes):** The size of the original uncompressed file in bytes. This is needed because the last byte of compressed data may contain padding.
4.  **Compressed Data Payload:** The sequence of Huffman codes packed into bytes.

## Usage

The application is run via the command line from the project's root directory.

### To Compress a File

1.  Run the script:
    ```sh
    python compress.py
    ```
2.  When prompted, press `c` and hit Enter.
3.  When prompted for the "File to compress:", enter the path to your file (e.g., `sample.txt` or `image.bmp`).

This will create a new file named `[original_filename].huf` in the same directory.

### To Decompress a File

1.  Run the script:
    ```sh
    python compress.py
    ```
2.  When prompted, press `d` and hit Enter.
3.  When prompted for the "File to decompress:", enter the path to the compressed file (e.g., `sample.txt.huf`).

This will create a new file named `[compressed_filename].orig` containing the original, decompressed content.
