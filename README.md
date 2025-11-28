# AI-Powered Python Debugger Sandbox

A local, autonomous debugging environment that detects and fixes both runtime exceptions and logical errors in Python code.

## 🚀 Features

- **Autonomous Repair Loop**: Automatically iterates on failing code, applying patches until the code runs successfully.
- **Logical Error Detection**: Goes beyond stack traces to identify subtle logical bugs (e.g., incorrect Tree Traversal orders).
- **Web Interface**: A clean, VS Code-like IDE for writing, running, and debugging code.
- **Detailed Analysis**: Provides step-by-step reasoning for every patch applied.
- **Safe Execution**: Runs user code in a temporary, isolated sandbox.

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/NishantDG-SST/SandBox.git
   cd SandBox/Python-Debugger-Sandbox
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Web Interface (Recommended)
Start the FastAPI server to use the graphical IDE:

```bash
python app.py
```
Open your browser to `http://localhost:8000`.

- **Run**: Executes the code and checks for errors (including logical ones).
- **Auto-Fix**: Enters the autonomous repair loop to fix detected issues.

### Command Line Demo
You can run the included demo script to see the debugger in action without the UI:

```bash
python demo_script.py
```

## 🧠 Capabilities

### 1. Runtime Error Fixing
The system can automatically fix common Python errors:
- `NameError` (undefined variables)
- `IndexError` (list bounds)
- `IndentationError`
- `SyntaxError` (missing colons)
- `TypeError` (string concatenation)
- `ZeroDivisionError`

### 2. Logical Error Detection
The system includes specialized heuristics to detect logical flaws that don't crash the program but produce incorrect results.

**Example: Preorder Traversal Bug**
If you write a `preorder` function that accidentally implements Inorder traversal (Left -> Root -> Right), the system will:
1. Detect the pattern mismatch.
2. Flag it as a **Logical Error**.
3. Automatically move the node visit operation to the correct position (Root -> Left -> Right).

## 📂 Project Structure

- `app.py`: Main FastAPI application and debugging logic.
- `static/index.html`: Frontend web interface.
- `demo_script.py`: CLI demonstration script.
- `requirements.txt`: Project dependencies.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
