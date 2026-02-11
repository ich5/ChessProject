# ♟️ AI Chess Bot

An AI-driven chess bot built using the **Python Lichess Bot framework**, combining a **neural network evaluation model** with a **minimax search algorithm** for intelligent move selection.

---

## 📌 Project Overview

This project implements a chess-playing agent that:

- Uses a **neural network** to evaluate board positions
- Uses **minimax search** with **alpha–beta pruning**
- Uses **iterative deepening** for time-controlled decision making
- Is trained using **supervised imitation learning**
- Uses **Stockfish** as an expert evaluator during training
- Integrates with the **Lichess Bot API**

The goal of the project is to demonstrate understanding of both **machine learning** and **classical AI search algorithms**.

---

## 🧠 How It Works

The system operates in three stages:

### 1️⃣ Opening Phase
The bot first checks for known opening moves to reduce computation.

### 2️⃣ Search Phase
If not in the opening:
- The bot runs **iterative deepening**
- Uses **minimax search**
- Applies **alpha–beta pruning**
- Explores future board positions up to a depth limit

### 3️⃣ Evaluation Phase
When the search reaches the depth limit:
- The board position is converted into **15 numerical features**
- These features are passed into a **trained neural network**
- The network outputs an evaluation score
- Minimax selects the move that leads to the best evaluated position

---

## 🏗 Architecture Overview

Board Position  
↓  
Feature Extraction (15 features)  
↓  
Neural Network Evaluation  
↓  
Minimax Search (Alpha–Beta Pruning)  
↓  
Best Move Selected  

---

## 🧩 Neural Network

- Implemented using **PyTorch**
- Input: 15 engineered chess features
- Output: 1 evaluation score
- Hidden layers: 256 → 128 → 64 neurons
- Activation: ReLU
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam

---

## 📊 Training Process

The neural network is trained using **supervised learning**:

1. Chess positions are extracted from ~50,000 real Lichess games.
2. **Stockfish** evaluates each position.
3. The neural network learns to approximate Stockfish’s evaluation.
4. Training runs for multiple epochs with mini-batch gradient descent.
5. The trained model is saved as `trained_nn_model.pth`.

This approach is known as **imitation learning**, because the model learns by imitating an expert evaluator.

---

## ⚙️ Technologies Used

- Python
- PyTorch
- Lichess Bot API
- Stockfish
- Minimax Search
- Alpha–Beta Pruning
- Iterative Deepening
- NumPy

---

## 📂 Project Structure
engines/bot/
│
├── main.py # Move selection entry point
├── minimax.py # Minimax + alpha-beta pruning
├── eval.py # Neural network evaluation
├── trained_nn_model.py # Training script
├── opening.py # Opening book logic
├── positions.py # Fallback evaluation tables


---

## 🚀 Running the Project

### Install Dependencies
pip install torch python-chess tqdm numpy

### Train the Model
python trained_nn_model.py

### Run the Bot
python -m engines.bot.main


---

## 📈 Strengths

- Combines **machine learning** with **classical AI search**
- Uses real-world chess data
- Efficient decision-making under time constraints
- Modular and extensible architecture

---

## ⚠️ Limitations

- Uses only 15 engineered features
- Limited search depth for performance reasons
- Imitation learning (not reinforcement learning)
- Model strength depends on training data size

---

## 🎓 Academic Context

This project was developed as part of an Artificial Intelligence course assignment to demonstrate:

- Supervised learning
- Neural networks
- Search algorithms
- Decision making under constraints
- Integration with external APIs

---

## 🏁 Summary

This chess bot demonstrates how a **learned evaluation model** can be integrated with a **classical search algorithm** to create an intelligent and explainable AI system.
