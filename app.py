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
    Returns: (new_code, diff, explanation) or (None, None, None) if no rule matched.
    """
    try:
        parsed = parse_traceback(stderr)
        if not parsed:
            return None, None, None
        stderr_low = stderr.lower()

        # 0) Replace literal backslash-t sequences (the user typed \t literally)
        if "\\t" in code:
            new_code = code.replace("\\t", "    ")
            diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
            explanation = "Replaced literal '\\t' with 4 spaces to fix unexpected character after line continuation."
            return new_code, diff, explanation

        # 1) NameError
        m = re.search(r"name '([^']+)' is not defined", stderr, re.IGNORECASE)
        if m:
            varname = m.group(1)
            default_val = "0"
            new_code = f"# Auto-added to fix NameError for '{varname}'\n{varname} = {default_val}\n" + code
            diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
            explanation = f"Defined variable '{varname}' at top with default {default_val}."
            return new_code, diff, explanation

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
                        guard_line = f"{leading_ws}if {loop_var} < len({arr_name}):"
                        inner_line = f"{leading_ws}    {stripped}"
                        new_lines = lines[:target_i] + [guard_line, inner_line] + lines[target_i + 1:]
                        new_code = "\n".join(new_lines)
                        diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
                        explanation = f"Added bounds-check `if {loop_var} < len({arr_name}):` around the indexing operation (safe heuristic)."
                        return new_code, diff, explanation

                # fallback: wrap the offending line but preserve indentation
                wrapped = lines[:target_i] + [
                    f"{leading_ws}try:",
                    f"{leading_ws}    {stripped}",
                    f"{leading_ws}except IndexError:",
                    f"{leading_ws}    print('IndexError handled')"
                ] + lines[target_i + 1:]
                new_code = "\n".join(wrapped)
                diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
                explanation = "Wrapped indexing operation in try/except with preserved indentation (safe fallback)."
                return new_code, diff, explanation

            # last-resort: wrap entire program
            new_code = "try:\n" + textwrap.indent(code, "    ") + "\nexcept IndexError:\n    print('IndexError handled')\n"
            diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
            return new_code, diff, "Wrapped entire program in try/except for IndexError (fallback)."

        # 3) IndentationError: fix real tabs
        if "indentationerror" in stderr_low:
            new_code = code.replace("\t", "    ")
            diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
            explanation = "Replaced tabs with 4 spaces to fix indentation."
            return new_code, diff, explanation

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
                            new_lines = lines.copy()
                            new_lines[lineno - 1] = ln + ":"
                            new_code = "\n".join(new_lines)
                            diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
                            explanation = f"Added missing ':' at line {lineno} (heuristic based on traceback)."
                            return new_code, diff, explanation
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
                new_code = "\n".join(new_lines)
                diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
                explanation = "Added missing colon(s) in control/function lines (fallback heuristic)."
                return new_code, diff, explanation


        # 5) TypeError print concat -> convert to comma-separated args
        if "typeerror" in stderr_low and "can only concatenate" in stderr_low:
            new_code = re.sub(r'print\((["\'].*?["\'])\s*\+\s*([a-zA-Z_][a-zA-Z0-9_]*)\)', r'print(\1, \2)', code)
            if new_code != code:
                diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
                explanation = "Converted string concatenation in print to comma-separated arguments."
                return new_code, diff, explanation

        # 6) ZeroDivisionError -> wrap operation
        if "zerodivisionerror" in stderr_low:
            lines = code.splitlines()
            for i, ln in enumerate(lines):
                if "/" in ln:
                    leading_ws = re.match(r"^(\s*)", ln).group(1)
                    stripped = ln[len(leading_ws):]
                    wrapped = lines[:i] + [
                        f"{leading_ws}try:",
                        f"{leading_ws}    {stripped}",
                        f"{leading_ws}except ZeroDivisionError:",
                        f"{leading_ws}    print('Division by zero handled')"
                    ] + lines[i + 1:]
                    new_code = "\n".join(wrapped)
                    diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
                    explanation = "Wrapped division operation in try/except to handle ZeroDivisionError (heuristic)."
                    return new_code, diff, explanation

        # No rule matched
        return None, None, None

    except Exception as e:
        # Defensive: log and return no patch so loop stops gracefully
        logger.exception("Internal patch-generator error: %s", e)
        return None, None, None

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
            success = True
            break

        new_code, diff, explanation = generate_patch_rule_based_v2(code, run_res["stderr"])
        if not new_code:
            # nothing to do or generator couldn't produce a safe fix
            break

        patches.append({"iteration": it, "diff": diff, "explanation": explanation})
        code = new_code

    return {"final_code": code, "patches": patches, "logs": logs, "success": success}

# ---------- API endpoints ----------
@app.post("/run")
async def run(req: RunRequest):
    # --- RUN ONLY MODE (minimal output) ---
    if req.run_only:
        result = run_python_code(req.code, timeout=req.timeout_seconds)
        return {
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "success": result.get("success", False)
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
