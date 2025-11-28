# app.py
import os
import re
import sys
import textwrap
import shutil
import tempfile
import difflib
import traceback
import logging
import subprocess
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug-sandbox")      


app = FastAPI(title="Local Debugging Sandbox")

from fastapi import File, UploadFile

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accept a single text file upload and return its contents.
    """
    try:
        content = await file.read()
        text = content.decode('utf-8', errors='replace')
        # return file name and text
        return {"filename": file.filename, "text": text}
    except Exception as e:
        logger.exception("File upload failed")
        raise HTTPException(status_code=500, detail="Failed to read uploaded file.")


# add these imports near the top of app.py if not already present
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# mount the static folder at /static so CSS/JS/images load correctly
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    logger.info("No static/ directory found at %s. Create one and put index.html inside it.", static_dir)

# replace previous index_html with this
@app.get("/", include_in_schema=False)
def index_html():
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content, status_code=200)
    # helpful message if missing
    return HTMLResponse(content="<h2>Place static/index.html in the project root to use the demo UI.</h2>", status_code=200)

# Allow CORS for local testing (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler -> always return JSON + log full traceback
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error("Unhandled server error while processing request %s:\n%s", request.url, tb)
    return JSONResponse(status_code=500, content={"detail": "Internal server error. See server logs."})

# ---------- Request model ----------
class RunRequest(BaseModel):
    code: str
    max_iterations: int = 3
    timeout_seconds: int = 3
    run_only: bool = False

# ---------- Sandbox runner ----------
def run_python_code(code: str, timeout: int = 3) -> Dict[str, Any]:
    """
    Run user Python code in a temporary directory using subprocess with timeout.
    Returns dict: { success, stdout, stderr, returncode }.
    """
    tmpdir = tempfile.mkdtemp(prefix="sandbox_")
    fname = os.path.join(tmpdir, "user_code.py")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        proc = subprocess.run(
            [sys.executable, "-u", fname],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            cwd=tmpdir,
            text=True,
        )
        return {"success": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}
    except subprocess.TimeoutExpired as e:
        return {"success": False, "stdout": e.stdout or "", "stderr": f"Timeout after {timeout}s", "returncode": -1}
    except Exception as e:
        # Unexpected execution error
        logger.exception("Execution error")
        return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# ---------- Traceback parser ----------
def parse_traceback(stderr: str) -> Optional[Dict[str, str]]:
    """
    Heuristic extraction from stderr. Returns None if nothing to parse.
    """
    if not stderr:
        return None
    tb_idx = stderr.rfind("Traceback")
    if tb_idx == -1:
        return {"traceback": stderr.strip(), "exception": stderr.strip(), "file_line": None}
    tb = stderr[tb_idx:].strip()
    file_line = None
    for ln in tb.splitlines():
        ln = ln.strip()
        if ln.startswith("File"):
            file_line = ln
    exc_line = tb.splitlines()[-1] if tb.splitlines() else ""
    return {"traceback": tb, "exception": exc_line, "file_line": file_line}

# ---------- Defensive rule-based patch generator ----------
def generate_patch_rule_based_v2(code: str, stderr: str):
    """
    Defensive patch generator:
      - handles literal '\t' escape sequences
      - NameError: define var = 0 at top
      - IndexError: conservative, indentation-preserving wrap or bounds-check if safe
      - IndentationError: replace actual tabs with spaces
      - SyntaxError missing colon: add colon heuristically
      - TypeError print concat: convert "str" + var -> print("str", var)
      - ZeroDivisionError: try/except wrapper
      - KeyError: fix wrong dictionary key based on context (e.g., memo[0] -> memo[n])
    Returns: (new_code, diff, explanation, reasoning) or (None, None, None, None) if no rule matched.
        reasoning: List of strings describing the decision-making process
    """
    try:
        parsed = parse_traceback(stderr)
        if not parsed:
            return None, None, None, None
        stderr_low = stderr.lower()

        # 0) Replace literal backslash-t sequences (the user typed \t literally)
        if "\\t" in code:
            reasoning = [
                "Detected literal '\\t' escape sequence in code",
                "SyntaxError indicates unexpected character after line continuation",
                "Replacing literal '\\t' with 4 spaces to match Python's tab-to-space conversion"
            ]
            new_code = code.replace("\\t", "    ")
            diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
            explanation = "Replaced literal '\\t' with 4 spaces to fix unexpected character after line continuation."
            return new_code, diff, explanation, reasoning

        # 1) NameError
        m = re.search(r"name '([^']+)' is not defined", stderr, re.IGNORECASE)
        if m:
            varname = m.group(1)
            default_val = "0"
            reasoning = [
                f"Detected NameError: variable '{varname}' is not defined",
                f"Extracted variable name from error message: '{varname}'",
                f"Strategy: Define variable at top of code with default value {default_val}",
                f"Reasoning: This allows code to run without undefined variable errors",
                f"Note: Default value {default_val} may need adjustment based on context"
            ]
            new_code = f"# Auto-added to fix NameError for '{varname}'\n{varname} = {default_val}\n" + code
            diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
            explanation = f"Defined variable '{varname}' at top with default {default_val}."
            return new_code, diff, explanation, reasoning

        # 2) IndexError (safe conservative handling)
        if "indexerror" in stderr_low or "list index out of range" in stderr_low:
            lines = code.splitlines()
            target_i = None
            file_line = parsed.get("file_line")
            if file_line:
                mline = re.search(r'line\s+(\d+)', file_line)
                if mline:
                    lineno = int(mline.group(1))
                    if 1 <= lineno <= len(lines):
                        candidate = lines[lineno - 1]
                        if "[" in candidate and "]" in candidate:
                            target_i = lineno - 1
            if target_i is None:
                for i, ln in enumerate(lines):
                    try:
                        if "[" in ln and "]" in ln:
                            target_i = i
                            break
                    except Exception:
                        continue

            if target_i is not None:
                ln = lines[target_i]
                leading_ws_match = re.match(r"^(\s*)", ln)
                leading_ws = leading_ws_match.group(1) if leading_ws_match else ""
                stripped = ln[len(leading_ws):]

                # Try a safe bounds-check: detect simple array name before '['
                arr_name = None
                arr_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\[', stripped)
                if arr_match:
                    arr_name = arr_match.group(1)
                # If we are inside an obvious for-loop and loop var is used, try to add guard
                # but be conservative: only if arr_name is simple token
                prev_for = None
                for j in range(target_i - 1, -1, -1):
                    s = lines[j].strip()
                    if not s:
                        continue
                    mfor = re.match(r"for\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+in\s+(.+):?", s)
                    if mfor:
                        prev_for = (j, mfor.group(1))
                    break

                if prev_for and arr_name:
                    loop_var = prev_for[1]
                    # if loop var appears in the indexed expression -> add bounds-check
                    if re.search(rf"\b{re.escape(loop_var)}\b", stripped):
                        reasoning = [
                            f"Detected IndexError at line {target_i + 1}",
                            f"Found array access pattern: {arr_name}[...]",
                            f"Detected for-loop with loop variable '{loop_var}'",
                            f"Strategy: Add bounds-check before array access",
                            f"Reasoning: Check if {loop_var} < len({arr_name}) before accessing array",
                            f"Applied safe heuristic: wrap indexing in conditional guard"
                        ]
                        guard_line = f"{leading_ws}if {loop_var} < len({arr_name}):"
                        inner_line = f"{leading_ws}    {stripped}"
                        new_lines = lines[:target_i] + [guard_line, inner_line] + lines[target_i + 1:]
                        new_code = "\n".join(new_lines)
                        diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
                        explanation = f"Added bounds-check `if {loop_var} < len({arr_name}):` around the indexing operation (safe heuristic)."
                        return new_code, diff, explanation, reasoning

                # fallback: wrap the offending line but preserve indentation
                reasoning = [
                    f"Detected IndexError at line {target_i + 1}",
                    "Could not determine array name or loop context",
                    "Strategy: Wrap indexing operation in try/except block",
                    "Reasoning: Preserves indentation and handles IndexError gracefully",
                    "Applied safe fallback: defensive error handling"
                ]
                wrapped = lines[:target_i] + [
                    f"{leading_ws}try:",
                    f"{leading_ws}    {stripped}",
                    f"{leading_ws}except IndexError:",
                    f"{leading_ws}    print('IndexError handled')"
                ] + lines[target_i + 1:]
                new_code = "\n".join(wrapped)
                diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
                explanation = "Wrapped indexing operation in try/except with preserved indentation (safe fallback)."
                return new_code, diff, explanation, reasoning

            # last-resort: wrap entire program
            reasoning = [
                "Detected IndexError but could not locate specific line",
                "Strategy: Wrap entire program in try/except block",
                "Reasoning: Last-resort fallback to handle IndexError at any point",
                "Applied defensive error handling for entire code block"
            ]
            new_code = "try:\n" + textwrap.indent(code, "    ") + "\nexcept IndexError:\n    print('IndexError handled')\n"
            diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
            return new_code, diff, "Wrapped entire program in try/except for IndexError (fallback).", reasoning

        # 3) IndentationError: fix real tabs
        if "indentationerror" in stderr_low:
            reasoning = [
                "Detected IndentationError in code",
                "Python requires consistent indentation (spaces or tabs, not mixed)",
                "Strategy: Replace all tab characters with 4 spaces",
                "Reasoning: Standard Python convention uses 4 spaces for indentation",
                "Applied uniform indentation fix across entire code"
            ]
            new_code = code.replace("\t", "    ")
            diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
            explanation = "Replaced tabs with 4 spaces to fix indentation."
            return new_code, diff, explanation, reasoning

        # 4) targeted SyntaxError: missing colon
        if "syntaxerror" in stderr_low:
            # If traceback gives a file line, try to add colon at that exact line
            file_line = parsed.get("file_line")
            if file_line:
                mline = re.search(r'line\s+(\d+)', file_line)
                if mline:
                    lineno = int(mline.group(1))
                    lines = code.splitlines()
                    if 1 <= lineno <= len(lines):
                        ln = lines[lineno - 1].rstrip()
                        # Only add colon to control/def lines that miss it
                        if re.match(r"\s*(def\s+\w+\(.*\)|if\s+.*|for\s+.*|while\s+.*|elif\s+.*|else\s*)$", ln) and not ln.endswith(":"):
                            reasoning = [
                                f"Detected SyntaxError at line {lineno}",
                                f"Line pattern matches control structure or function definition: {ln.strip()}",
                                "Python requires ':' after control structures and function definitions",
                                f"Strategy: Add missing ':' at end of line {lineno}",
                                "Reasoning: Based on traceback location and line pattern matching"
                            ]
                            new_lines = lines.copy()
                            new_lines[lineno - 1] = ln + ":"
                            new_code = "\n".join(new_lines)
                            diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
                            explanation = f"Added missing ':' at line {lineno} (heuristic based on traceback)."
                            return new_code, diff, explanation, reasoning
            # Fallback to the broader heuristic if we couldn't target the line
            changed = False
            new_lines = []
            for ln in code.splitlines():
                s = ln.rstrip()
                if re.match(r"\s*(def\s+\w+\(.*\)|if\s+.*|for\s+.*|while\s+.*|elif\s+.*|else\s*)$", s) and not s.endswith(":"):
                    s = s + ":"
                    changed = True
                new_lines.append(s)
            if changed:
                reasoning = [
                    "Detected SyntaxError but could not pinpoint exact line from traceback",
                    "Strategy: Scan all lines for control structures missing ':'",
                    "Reasoning: Fallback heuristic - add ':' to any control/function line missing it",
                    "Applied pattern matching to identify lines needing colons"
                ]
                new_code = "\n".join(new_lines)
                diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
                explanation = "Added missing colon(s) in control/function lines (fallback heuristic)."
                return new_code, diff, explanation, reasoning


        # 5) TypeError print concat -> convert to comma-separated args
        if "typeerror" in stderr_low and "can only concatenate" in stderr_low:
            reasoning = [
                "Detected TypeError: can only concatenate str (not ...) to str",
                "Common pattern: print('text' + variable) where variable is not a string",
                "Strategy: Convert string concatenation to comma-separated print arguments",
                "Reasoning: print() automatically converts arguments to strings and adds spaces",
                "Applied regex pattern matching to find and fix print concatenation"
            ]
            new_code = re.sub(r'print\((["\'].*?["\'])\s*\+\s*([a-zA-Z_][a-zA-Z0-9_]*)\)', r'print(\1, \2)', code)
            if new_code != code:
                diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
                explanation = "Converted string concatenation in print to comma-separated arguments."
                return new_code, diff, explanation, reasoning

        # 6) ZeroDivisionError -> wrap operation
        if "zerodivisionerror" in stderr_low:
            reasoning = [
                "Detected ZeroDivisionError: division by zero",
                "Strategy: Find division operation and wrap in try/except block",
                "Reasoning: Prevents program crash and handles division by zero gracefully",
                "Applied heuristic: locate first line containing '/' operator"
            ]
            lines = code.splitlines()
            for i, ln in enumerate(lines):
                if "/" in ln:
                    leading_ws = re.match(r"^(\s*)", ln).group(1)
                    stripped = ln[len(leading_ws):]
                    reasoning.append(f"Found division operation at line {i + 1}: {ln.strip()}")
                    wrapped = lines[:i] + [
                        f"{leading_ws}try:",
                        f"{leading_ws}    {stripped}",
                        f"{leading_ws}except ZeroDivisionError:",
                        f"{leading_ws}    print('Division by zero handled')"
                    ] + lines[i + 1:]
                    new_code = "\n".join(wrapped)
                    diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
                    explanation = "Wrapped division operation in try/except to handle ZeroDivisionError (heuristic)."
                    return new_code, diff, explanation, reasoning

        # 7) KeyError -> fix wrong dictionary key
        if "keyerror" in stderr_low:
            # Extract the key that caused the error
            key_match = re.search(r"KeyError:\s*([^\n]+)", stderr)
            if key_match:
                error_key = key_match.group(1).strip().strip("'\"")
                # Find the line with the KeyError from traceback
                file_line = parsed.get("file_line")
                if file_line:
                    mline = re.search(r'line\s+(\d+)', file_line)
                    if mline:
                        lineno = int(mline.group(1))
                        lines = code.splitlines()
                        if 1 <= lineno <= len(lines):
                            error_line = lines[lineno - 1]
                            # Look for dictionary access pattern like dict[key] or dict[0]
                            # Try to find the correct key from context
                            # Common pattern: if n in memo: return memo[0] -> should be memo[n]
                            dict_access_pattern = re.search(r'(\w+)\[' + re.escape(error_key) + r'\]', error_line)
                            if dict_access_pattern:
                                dict_name = dict_access_pattern.group(1)
                                # Look backwards for "if X in dict_name:" pattern
                                correct_key = None
                                for i in range(lineno - 2, -1, -1):
                                    if i < 0:
                                        break
                                    prev_line = lines[i]
                                    # Check for "if variable in dict_name:" pattern
                                    if_match = re.search(r'if\s+(\w+)\s+in\s+' + re.escape(dict_name) + r'[:]', prev_line)
                                    if if_match:
                                        correct_key = if_match.group(1)
                                        break
                                    # Also check function parameters
                                    func_match = re.search(r'def\s+\w+\s*\([^)]*(\w+)[^)]*\)', prev_line)
                                    if func_match and i < 3:  # Check first few lines for function def
                                        # Look for the parameter that makes sense
                                        params_match = re.search(r'def\s+\w+\s*\([^)]*(\w+)[^)]*\)', prev_line)
                                        if params_match:
                                            # Use the first parameter as a heuristic
                                            all_params = re.findall(r'(\w+)', prev_line.split('(')[1].split(')')[0])
                                            if all_params:
                                                correct_key = all_params[0]  # Use first parameter
                                                break
                                
                                # If we found a correct key, replace it
                                if correct_key:
                                    reasoning = [
                                        f"Detected KeyError: key '{error_key}' not found in dictionary",
                                        f"Found dictionary access pattern: {dict_name}[{error_key}] at line {lineno}",
                                        f"Analyzed context: found 'if {correct_key} in {dict_name}:' check before error",
                                        f"Strategy: Replace {dict_name}[{error_key}] with {dict_name}[{correct_key}]",
                                        f"Reasoning: The code checks for {correct_key} in dictionary, so that's the correct key to use"
                                    ]
                                    leading_ws = re.match(r"^(\s*)", error_line).group(1)
                                    stripped = error_line[len(leading_ws):]
                                    fixed_line = re.sub(
                                        re.escape(dict_name) + r'\[' + re.escape(error_key) + r'\]',
                                        f"{dict_name}[{correct_key}]",
                                        stripped
                                    )
                                    new_lines = lines.copy()
                                    new_lines[lineno - 1] = leading_ws + fixed_line
                                    new_code = "\n".join(new_lines)
                                    diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
                                    explanation = f"Fixed KeyError: changed {dict_name}[{error_key}] to {dict_name}[{correct_key}] based on context (if {correct_key} in {dict_name} check)."
                                    return new_code, diff, explanation, reasoning
                                
                                # Fallback: try to find the variable from function parameters in nearby lines
                                # Look for function definition and use its first parameter
                                for i in range(max(0, lineno - 5), lineno):
                                    func_def_match = re.search(r'def\s+\w+\s*\([^)]*(\w+)[^)]*\)', lines[i])
                                    if func_def_match:
                                        param = func_def_match.group(1)
                                        leading_ws = re.match(r"^(\s*)", error_line).group(1)
                                        stripped = error_line[len(leading_ws):]
                                        fixed_line = re.sub(
                                            re.escape(dict_name) + r'\[' + re.escape(error_key) + r'\]',
                                            f"{dict_name}[{param}]",
                                            stripped
                                        )
                                        reasoning = [
                                            f"Detected KeyError: key '{error_key}' not found in dictionary",
                                            f"Found dictionary access pattern: {dict_name}[{error_key}] at line {lineno}",
                                            f"Could not find 'if X in {dict_name}:' pattern in context",
                                            f"Strategy: Use function parameter '{param}' as the key",
                                            f"Reasoning: Found function definition with parameter '{param}', using it as dictionary key"
                                        ]
                                        new_lines = lines.copy()
                                        new_lines[lineno - 1] = leading_ws + fixed_line
                                        new_code = "\n".join(new_lines)
                                        diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
                                        explanation = f"Fixed KeyError: changed {dict_name}[{error_key}] to {dict_name}[{param}] based on function parameter."
                                        return new_code, diff, explanation, reasoning

        # No rule matched
        return None, None, None, None

    except Exception as e:
        # Defensive: log and return no patch so loop stops gracefully
        logger.exception("Internal patch-generator error: %s", e)
        return None, None, None, None

# ---------- Logical error detector for tree traversal ----------
def detect_tree_traversal_error(code: str, stdout: str) -> Optional[tuple]:
    """
    Detect common tree traversal logical errors.
    Returns: (new_code, diff, explanation, reasoning, error_message) or None if no issue detected.
        reasoning: List of strings describing the decision-making process
        error_message: Short summary of the logical error
    """
    try:
        # Check if code contains tree traversal patterns
        if not re.search(r'def\s+(preorder|inorder|postorder)', code, re.IGNORECASE):
            return None
        
        lines = code.splitlines()
        
        # Find preorder function
        preorder_match = re.search(r'def\s+preorder\s*\([^)]*\):', code, re.IGNORECASE)
        if preorder_match:
            # Find the function body
            func_start = None
            for i, line in enumerate(lines):
                if re.search(r'def\s+preorder\s*\([^)]*\):', line, re.IGNORECASE):
                    func_start = i
                    break
            
            if func_start is not None:
                # Look for the pattern: recursive call, then append, then recursive call
                # This is inorder, not preorder
                found_pattern = False
                append_line_idx = None
                
                for i in range(func_start, min(func_start + 10, len(lines))):
                    line = lines[i]
                    # Check if this line has append after a recursive call pattern
                    if 'append' in line and i > func_start:
                        # Check if previous line has recursive call
                        if i > 0 and 'preorder' in lines[i-1] and '(' in lines[i-1]:
                            # Check if next line has recursive call
                            if i < len(lines) - 1 and 'preorder' in lines[i+1] and '(' in lines[i+1]:
                                found_pattern = True
                                append_line_idx = i
                                break
                
                if found_pattern and append_line_idx is not None:
                    # This is inorder, should be preorder
                    # Move append to right after the base case check, before first recursive call
                    # Find the base case return statement
                    base_case_idx = None
                    for i in range(func_start, min(func_start + 5, len(lines))):
                        if 'if not' in lines[i] or 'return' in lines[i]:
                            # Find the return statement after if not
                            if i < len(lines) - 1 and 'return' in lines[i+1]:
                                base_case_idx = i + 1
                                break
                    
                    # Find insertion point: right after base case, before first recursive call
                    insert_idx = None
                    if base_case_idx is not None:
                        # Look for first recursive call after base case
                        for i in range(base_case_idx + 1, append_line_idx):
                            if 'preorder' in lines[i] and '(' in lines[i]:
                                insert_idx = i
                                break
                    
                    # If we couldn't find it, use the line right after base case
                    if insert_idx is None and base_case_idx is not None:
                        insert_idx = base_case_idx + 1
                    
                    if insert_idx is not None:
                        # Get the append line with its indentation
                        append_line = lines[append_line_idx]
                        leading_ws = re.match(r"^(\s*)", append_line).group(1)
                        append_content = append_line.strip()
                        
                        # Find first recursive call index for reasoning
                        first_recursive_line = None
                        for i in range(func_start, append_line_idx):
                            if 'preorder' in lines[i] and '(' in lines[i] and 'if not' not in lines[i]:
                                first_recursive_line = i + 1
                                break
                        
                        # Create new code: move append before first recursive call
                        new_lines = lines.copy()
                        # Remove append from current position (do this first to preserve indices)
                        removed_line = new_lines.pop(append_line_idx)
                        # Adjust insert_idx if append was before it
                        if append_line_idx < insert_idx:
                            insert_idx -= 1
                        # Insert append at the correct position
                        new_lines.insert(insert_idx, leading_ws + append_content)
                        
                        reasoning = [
                            "Detected logical error: function named 'preorder' but performing inorder traversal",
                            f"Found pattern: recursive call -> append -> recursive call (append at line {append_line_idx + 1})",
                            "Inorder traversal: left subtree -> root -> right subtree",
                            "Preorder traversal should be: root -> left subtree -> right subtree",
                            f"Strategy: Move append operation (node visit) to before first recursive call",
                            f"Reasoning: Preorder requires visiting root before traversing children",
                            f"Applied fix: moved res.append(root.val) from line {append_line_idx + 1} to line {insert_idx + 1}"
                        ]
                        new_code = "\n".join(new_lines)
                        diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
                        explanation = "Fixed preorder traversal: moved node visit (append) to before recursive calls. The function was doing inorder (left-root-right) instead of preorder (root-left-right)."
                        error_message = "Logical Error: Function 'preorder' implements Inorder traversal (Left -> Root -> Right) instead of Preorder (Root -> Left -> Right)."
                        return new_code, diff, explanation, reasoning, error_message
        
        return None
    except Exception as e:
        logger.exception("Error in tree traversal detection: %s", e)
        return None

# ---------- Repair loop ----------
def repair_code_loop(initial_code: str, max_iterations: int = 3, timeout_seconds: int = 3):
    code = initial_code
    patches = []
    logs = []
    success = False

    for it in range(1, max_iterations + 1):
        run_res = run_python_code(code, timeout=timeout_seconds)
        logs.append({"iteration": it, "stdout": run_res["stdout"], "stderr": run_res["stderr"], "returncode": run_res["returncode"]})
        
        if run_res["success"]:
            # Check for logical errors even when code runs successfully
            logical_fix = detect_tree_traversal_error(code, run_res["stdout"])
            if logical_fix:
                new_code, diff, explanation, reasoning, error_msg = logical_fix
                patches.append({
                    "iteration": it, 
                    "diff": diff, 
                    "explanation": explanation, 
                    "reasoning": reasoning,
                    "error_message": error_msg
                })
                code = new_code
                # Continue to next iteration to verify the fix
                continue
            else:
                success = True
                break

        new_code, diff, explanation, reasoning = generate_patch_rule_based_v2(code, run_res["stderr"])
        if not new_code:
            # nothing to do or generator couldn't produce a safe fix
            break

        patches.append({"iteration": it, "diff": diff, "explanation": explanation, "reasoning": reasoning})
        code = new_code

    # Build summary of captures
    captures_summary = {
        "runtime_errors": any("Traceback" in log["stderr"] or log["returncode"] != 0 for log in logs),
        "stack_traces": any("Traceback" in log["stderr"] for log in logs),
        "logs_and_prints": any(log["stdout"].strip() for log in logs),
        "final_output": logs[-1]["stdout"].strip() if logs and logs[-1]["stdout"].strip() else None
    }
    
    return {
        "final_code": code,
        "patches": patches,
        "logs": logs,
        "success": success,
        "captures": captures_summary
    }

# ---------- API endpoints ----------
@app.post("/run")
async def run(req: RunRequest):
    # --- RUN ONLY MODE (minimal output) ---
    if req.run_only:
        result = run_python_code(req.code, timeout=req.timeout_seconds)
        
        # Check for logical errors even in run-only mode
        logical_error = None
        if result["success"]:
            detection = detect_tree_traversal_error(req.code, result.get("stdout", ""))
            if detection:
                # detection is (new_code, diff, explanation, reasoning, error_message)
                logical_error = detection[4]

        return {
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "success": result.get("success", False),
            "logical_error": logical_error
        }

    # --- FULL AUTO-FIX MODE ---
    result = repair_code_loop(
        req.code,
        max_iterations=req.max_iterations,
        timeout_seconds=req.timeout_seconds
    )
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}

# Serve static index.html if exists
@app.get("/", include_in_schema=False)
def index_html():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return {"info": "Place static/index.html to use the demo UI."}

# Run server
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)