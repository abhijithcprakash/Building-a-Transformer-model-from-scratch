# Language Model Implementation from Scratch: A Transformer-Based Approach

## Introduction

This repository presents an implementation of a language model based on the **Transformer architecture** introduced in the seminal paper *"Attention is All You Need"* by Vaswani et al. (2017). The goal of this project is to provide a clean and minimal implementation of an autoregressive language model using PyTorch.

By going through this project, the reader will gain a deeper understanding of the Transformer model, the inner workings of attention mechanisms, and the process of training a model for text generation. The focus of this implementation is on the core principles of the architecture and building it from scratch rather than relying on pre-built libraries like Hugging Face's `transformers`.

### The Research Paper: "Attention is All You Need"

The Transformer model introduced in the paper *"Attention is All You Need"* has revolutionized the field of natural language processing (NLP). It is based on the **self-attention mechanism**, which allows the model to weigh the importance of different words in a sentence, regardless of their position. This differs from traditional models like RNNs and LSTMs, which process text sequentially and struggle with long-range dependencies.

Key points from the paper:
- **Self-Attention**: The ability to consider all other words in the input sequence when processing each word.
- **Parallelization**: Unlike RNNs and LSTMs, the Transformer does not require sequential data processing. This allows for better parallelization during training, leading to faster training times.
- **Positional Encoding**: Since the Transformer does not have a built-in understanding of word order (unlike RNNs), positional encodings are added to give the model information about the position of words in the sequence.
- **Scalability**: The Transformer model scales well to large datasets and can be trained on distributed systems, which is why it became the foundation for models like BERT, GPT, and T5.

The self-attention mechanism and parallelization made the Transformer the architecture of choice for modern NLP tasks.

---

## Why Implementing from Scratch is Important

In the age of advanced libraries like Hugging Face's `transformers`, it's easy to overlook the value of implementing models from scratch. However, there are several reasons why it's beneficial to occasionally build models like the Transformer from the ground up:

1. **Deep Understanding**:
   - Implementing a model from scratch helps you gain a deeper understanding of its components, such as the self-attention mechanism, the encoder-decoder architecture, and how training flows.
   - You get to handle the low-level details like batching, padding, masking, and loss calculation, which are often abstracted away in high-level libraries.

2. **Customization**:
   - When you implement a model yourself, you can easily experiment with different hyperparameters, loss functions, and optimizations.
   - It allows you to add custom layers, build unique architectures, or experiment with new techniques without waiting for a library update.

3. **Learning Experience**:
   - Building from scratch gives you practical experience with core concepts like backpropagation, gradient descent, and optimization.
   - It forces you to think critically about how models work internally and helps you troubleshoot potential issues more effectively.

4. **Better Debugging**:
   - When using pre-built models, it can be difficult to trace errors or understand how specific parts of the model work. By writing the model yourself, you can debug issues much more easily.
   - You'll be more equipped to fine-tune the model and make improvements as you understand the inner workings more deeply.

---

## Project Overview

This project implements a Transformer-based autoregressive language model for text generation. The model is trained using a small corpus of text and can generate new text based on a given prompt.

### Key Components

1. **Model Architecture**:
   - The architecture is based on the **Transformer Decoder**, which is used for autoregressive language modeling (like GPT).
   - The model is composed of the following components:
     - **Token Embedding**: Converts token indices into dense vector representations.
     - **Positional Encoding**: Adds information about the position of each token in the sequence.
     - **Self-Attention Layer**: The key mechanism in the Transformer, allowing the model to focus on different parts of the input sequence.
     - **Feedforward Layers**: Non-linear transformations that process the information passed through the attention layers.
     - **Output Layer**: A final layer that predicts the next token in the sequence.

2. **Training**:
   - The model is trained using the **cross-entropy loss** function, which compares the model’s predictions to the target sequence.
   - **Gradient clipping** is applied to prevent exploding gradients, which can be a common issue in deep networks like the Transformer.
   - The model is optimized using the **Adam optimizer**.

3. **Text Generation**:
   - After training, the model can generate text one token at a time, conditioned on previous tokens. This is done in an autoregressive manner, where each predicted token is appended to the input sequence for generating the next token.

4. **Tokenizer**:
   - A simple **character-level tokenizer** is implemented for tokenizing input text into indices.
   - It supports padding and character-to-token conversions.

### File Structure

```
├── README.md
├── model.py           # Model classes for TokenEmbedding, PositionalEncoding, LanguageModel, etc.
├── trainer.py         # The Trainer class for managing training loops
├── generator.py       # The Generator class for text generation
├── utils.py           # Helper functions for padding and tokenization
├── data.py            # Functions for preparing training data
├── trained_model/     # Directory where the model checkpoint is saved
├── train.py           # Main script for training the model
└── requirements.txt   # List of dependencies
# EVERYTHING IS KEPT IN A SINGLE .py FILE
```

---

## Installation

To run this project, you need to have Python 3.6 or higher installed along with the required libraries. You can install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### Requirements
- **PyTorch**: Deep learning framework used for building and training the model.
- **NumPy**: Used for numerical operations.
- **Matplotlib**: For plotting the training loss.
- **Random**: For shuffling and data manipulation.

---

## Usage

1. **Training the Model**:
   - To start training, run the `train.py` script:
   ```bash
   python train.py
   ```

2. **Text Generation**:
   - After training, you can generate text by running the following command:
   ```bash
   python generate.py --prompt "cats"
   ```

3. **Saving and Loading Checkpoints**:
   - The model supports saving and loading checkpoints using the `save_checkpoint` and `load_checkpoint` methods.

---

## Results

During training, the model's loss is printed for each epoch, and a plot of the loss over time is generated. This can be used to track the model's performance and ensure that it is converging.

After training, the model can generate text given a prompt. For example, if you prompt the model with "cats", it may generate text such as:

```
cats rule the world. They are the kings of the animal kingdom, and their reign is unchallenged. Cats are known for their agility, speed, and independence. They rule the world in silence, and their paws leave no trace.
```

---

## Conclusion

This repository provides an implementation of a Transformer-based autoregressive language model from scratch, inspired by the "Attention is All You Need" paper. The project emphasizes the importance of understanding the model architecture and the training process by building it from the ground up. Whether you're looking to learn more about the Transformer model or wanting to experiment with your own implementations, this project provides a solid foundation for exploring NLP and text generation.

---
