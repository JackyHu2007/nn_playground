![Neural Network Playground](assets/screenshot-main.jpeg)
# Neural Network Playground (Python & Web)

A full-stack, from-scratch rewrite of the [TensorFlow Playground](https://playground.tensorflow.org/) (based on the earth science fork [Rocky Playground](https://github.com/agilescientific/rocky-playground)).

Unlike the original which runs entirely in the browser using TensorFlow.js, this version moves all neural network computation to a **Python backend** (using pure **NumPy**), communicating with a lightweight vanilla JS frontend via **WebSockets** in real-time.

This is not just a toy—it is a rigorous, interactive laboratory designed to let you visualize, tweak, and understand advanced deep learning architectures and their underlying mathematics.

![Tutorial](assets/screenshot-tutorial.jpeg)
*Rigorous math tutorials rendered with KaTeX.*

## ✨ Core Features

###  Advanced Network Architecture
*   **Mixture of Experts (MoE)**: Build MoE blocks with up to 16 experts. Supports both **Top-k** (sparse) and **Soft** (dense) routing.
    *   **Deep Experts**: Each expert can have its own independent stack of 1–8 hidden layers.
    *   **Load Balancing**: Built-in Switch-Transformer style auxiliary loss (`aux_coef`) to prevent expert collapse.
    *   **Router Temperature**: Control the sharpness of the gating probabilities.
*   **Normalization**: LayerNorm and RMSNorm, with selectable **pre** or **post** placement.
*   **Residual Connections**: Add arbitrary skip-links between layers of matching dimensions to study gradient flow.
*   **Gradient Clipping**: Prevent exploding gradients with selectable **Clip by Value** or **Clip by Norm** methods directly from the top control bar.
*   **13 Activation Functions**: From classics (Sigmoid, Tanh, ReLU) to modern ones (GELU, Swish, Mish, SELU) and even high-frequency (Sine).

###  Training & Regularization
*   **Optimizers**: Standard SGD, plus Momentum, RMSProp, and Adam.
*   **Hard L1 Regularization**: Witness sparsity in action—weights that cross zero are permanently frozen ("dead").
*   **Online Data**: Resample the training set from probability distributions at every epoch to test true generalization.
*   **Automatic Python Code Export**: Instantly compile your network (including MoE and LayerNorm logic) into a standalone, dependency-free Python function.

###  Datasets & Visualization
*   **Huge Dataset Zoo**: 16 synthetic binary sets + 2 real-world (rocks, poroperm), 40 multiclass sets (including fractal and high-frequency hell-tier sets like *Mandala* and *Dense Spiral*), 11 synthetic regression sets + 2 real-world (porosity, DTS).
*   **Custom Data Upload**: Upload your own JSON `[{x, y, label}, ...]`, and the backend will automatically infer the problem type and class count.
*   **Interactive Tutorials**: A built-in tutorial page (`/tutorial.html`) featuring:
    *   **Theory Tab**: Rigorous mathematical derivations (Jacobians, backprop, MoE routing math) rendered with KaTeX.
    *   **Usage Tab**: Practical guides on how to read the heatmaps, debug dead neurons, and tune MoE routing.
*   **Deep Inspection**: Click any neuron or router node to zoom in and see its specific decision boundary.

###  Improvements based on the original version
*   **Correct ELU / Swish derivatives** — verified against finite differences
  in `tests/test_core.py` (the original playground's implementation is not
  documented here; this backend's math is self-tested).
*   **Non-overlapping train/test splits** for all real-world datasets — the
  original implementation had overlapping splits for `poroperm` and
  `porosity`, which made "test loss" meaningless. See `data/loaders.py`.
*   **Optimizer state is cleared on architecture change** — Adam/Momentum
  caches are reset when the network is rebuilt, so changing layer widths
  does not throw a shape-mismatch `ValueError`.

## 📂 Project Structure

```text
nn_playground/
├── backend/                 # Python FastAPI + NumPy Backend
│   ├── main.py              # App entry point
│   ├── api/                 # REST, Custom Upload, and WebSocket routers
│   ├── core/                # Pure NumPy math (NN, MoE, Optimizers, Gradient Clipping)
│   ├── data/                # 50+ Synthetic generators and JSON loaders
│   └── session/             # Training state and session manager
├── data/                    # Original JSON datasets (rocks, porosity, etc.)
├── frontend/                # Vanilla JS + D3.js Frontend
│   ├── index.html           # Main Playground
│   ├── tutorial.html        # Math & Usage Tutorials (KaTeX)
│   ├── css/styles.css
│   └── js/                  # API client, UI bindings, Canvas/SVG renderers
├── tests/                   # Pytest suite (math, API, MoE, and gradient checks)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Requirements
- Python 3.10+
- A modern web browser (Chrome, Firefox, Safari, Edge)

### 2. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/JackyHu2007/nn_playground.git
cd nn_playground
pip install -r requirements.txt
```

*(Make sure you have `uvicorn[standard]` installed to support WebSockets.)*

### 3. Run the Application
Start the FastAPI backend server:

```bash
uvicorn backend.main:app --reload
```

### 4. Open in Browser
Navigate to **[http://127.0.0.1:8000](http://127.0.0.1:8000)** to start tinkering! 
*Click the **"Tutorial"** button in the top right corner to read the math and user guide.*

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, WebSockets, NumPy, Pydantic
- **Frontend**: HTML5, CSS3, Vanilla JavaScript (ES Modules), KaTeX, **Prism.js**
- **Visualization**: D3.js (v7), HTML5 Canvas, SVG
- **Testing**: Pytest

## 🧪 Running Tests

To verify the mathematical correctness of the neural network (including finite-difference gradient checking, MoE tensor routing, and gradient clipping limits):

```bash
python -m pytest tests/ -v
```

## ⚠️ Known Limitations

- **Single-process session store.** Sessions live in an in-memory dict
  (`backend/session/manager.py`); horizontal scaling would require Redis
  or similar. This is intentional for a self-contained playground.
- **No persistent storage.** Custom datasets and saved sessions expire
  after 1 hour of inactivity, or on server restart. Use the **Save** button
  (top-right) to persist a configuration to disk.
- **One WebSocket owner per session.** `backend/api/websocket.py` refuses to
  rejoin a session that already has a live connection, to prevent two
  trainers from mutating the same `TrainingSession` under different locks.
- **Custom data must be roughly in `[-6, 6]²`.** The heat-map only renders
  that range; out-of-range points are accepted but invisible.
- **No GPU.** All math is NumPy, single-threaded per request. Suitable for
  ≤ 50 000 points and ≤ 12 hidden layers.

## 🙏 Credits

- Original concept and UI design by [Daniel Smilkov and Shan Carter](https://playground.tensorflow.org/).
- Earth science dataset integration and modifications by [Agile Scientific](https://agilescientific.com/) (Rocky Playground).
- Extended architecture (MoE, Normalization, Tutorials, Gradient Clipping) and Python backend implemented from scratch for this project.
- Some tests, documentation, and UI refinements were drafted with the assistance of **DeepSeek** (large language model), then reviewed, tested, and integrated by the author.

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

