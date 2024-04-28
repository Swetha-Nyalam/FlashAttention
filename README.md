# Flash Attention Implementation

Welcome to the Flash Attention Implementation project! This repository showcases my implementation of the Flash Attention algorithm, a powerful attention mechanism widely used in machine learning tasks. Below are the key technical aspects of this project:

## Introduction

Flash Attention is known for its efficiency and effectiveness in various machine learning tasks. In this project, I've implemented Flash Attention with specific parameters:

- **N (input sequence length)**: 3072
- **d_model (model size)**: 128
- **M (SRAM size M)**: 1536

## Key Technical Components

### 1. Implementation of Flash Attention Algorithm
- I've implemented the Flash Attention algorithm from scratch, understanding its core principles and intricacies.

### 2. Understanding Q, K, V Matrices
- Delved deep into the concept of Query (Q), Key (K), and Value (V) matrices, crucial for computation in attention mechanisms.

### 3. Softmax Operation
- Explored the softmax operation applied to attention scores, essential for determining the importance of input sequence elements.

### 4. Normalization Methods
- Studied various normalization methods to ensure stable and efficient training of the Flash Attention model.

### 5. Importance of Flash Attention
- Investigated the advantages of Flash Attention over conventional methods, highlighting its potential for enhancing performance in machine learning tasks.

## Conclusion

This project represents a significant technical milestone in my journey of learning machine learning concepts. By implementing and exploring Flash Attention, I've gained valuable insights into its technical intricacies and potential applications. I invite you to explore the code and documentation to gain a deeper understanding of Flash Attention and its implementation details.

## Blogs that helped me understand and implement

* [Flash attention paper](https://lnkd.in/gHTCNkYa)
* [Flash Attention Explained](https://lnkd.in/gHTCNkYa) by Jalammar [illustrated-transformer.github.io](https://jalammar.github.io/illustrated-transformer/)
* [Understanding Q, K, V in Transformer Self Attention](https://medium.com/analytics-vidhya/understanding-q-k-v-in-transformer-self-attention-9a5eddaa5960) by Analytics Vidhya
* [Understanding Flash Attention with ELI5](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad) by Gordical Ekşa
