
import sys
import os
import json

# Add current directory to path so we can import app
sys.path.append(os.getcwd())

from app import repair_code_loop

user_code = """class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def preorder(root, res):
    if not root:
        return
    
    preorder(root.left, res)
    res.append(root.val)
    preorder(root.right, res)

root = Node(1)
root.left = Node(2)
root.right = Node(3)

res = []
preorder(root, res)
print(res)
"""

print("--- Running Autonomous Debugger Demo ---")
print("Input Code:")
print(user_code)
print("-" * 30)

result = repair_code_loop(user_code)

print("\n--- Results ---")
print(f"Success: {result['success']}")
print(f"Final Code:\n{result['final_code']}")

print("\n--- Patches Applied ---")
for patch in result['patches']:
    print(f"Iteration {patch['iteration']}:")
    print(f"Explanation: {patch['explanation']}")
    print(f"Reasoning: {json.dumps(patch['reasoning'], indent=2)}")
    print(f"Diff:\n{patch['diff']}")

print("\n--- Logs ---")
for log in result['logs']:
    print(f"Iteration {log['iteration']}:")
    print(f"Stdout: {log['stdout'].strip()}")
    print(f"Stderr: {log['stderr'].strip()}")
