# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess, tempfile, os, shutil, textwrap, sys, difflib, re
from typing import List, Optional, Dict, Any
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Local Debugging Sandbox")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request / Response models ----------
class RunRequest(BaseModel):
    code: str
    max_iterations: int = 3
    timeout_seconds: int = 3

class PatchEntry(BaseModel):
    iteration: int
    diff: str
    explanation: str

# ---------- Sandbox runner ----------
def run_python_code(code: str, timeout: int = 3) -> Dict[str, Any]:
    """
    Writes code to temporary file and executes it using the current Python interpreter.
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
        return {
            "success": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired as e:
        return { "success": False, "stdout": e.stdout or "", "stderr": f"Timeout after {timeout}s", "returncode": -1 }
    except Exception as e:
        return { "success": False, "stdout": "", "stderr": str(e), "returncode": -1 }
    finally:
        # cleanup
        shutil.rmtree(tmpdir, ignore_errors=True)

# ---------- Traceback parser ----------
def parse_traceback(stderr: str) -> Optional[Dict[str, str]]:
    """
    Heuristic extraction from stderr. Returns None if no error-like content.
    """
    if not stderr:
        return None
    tb_idx = stderr.rfind("Traceback")
    if tb_idx == -1:
        # fallback: provide exception line as message
        return {"traceback": stderr.strip(), "exception": stderr.strip(), "file_line": None}
    tb = stderr[tb_idx:].strip()
    # find last file/line context
    file_line = None
    for ln in tb.splitlines():
        ln = ln.strip()
        if ln.startswith("File"):
            file_line = ln
    exc_line = tb.splitlines()[-1] if tb.splitlines() else ""
    return {"traceback": tb, "exception": exc_line, "file_line": file_line}

# ---------- Rule-based patch generator (returns new_code, diff, explanation) ----------
def generate_patch_rule_based_v2(code: str, stderr: str):
    """
    Defensive, improved rule-based generator.
    - Safely handles literal "\t"
    - Preserves indentation when wrapping indexing lines
    - Prefers bounds-check if it can detect loop variable and array name, else uses try/except with preserved indentation
    - Never raises internal exceptions (returns (None,None,None) on unexpected internal errors)
    """
    try:
        parsed = parse_traceback(stderr)
        if not parsed:
            return None, None, None
        stderr_low = stderr.lower()

        # Handle literal backslash-t sequences: convert "\\t" -> 4 spaces
        if "\\t" in code:
            new_code = code.replace("\\t", "    ")
            diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
            explanation = "Replaced literal '\\t' with 4 spaces to fix unexpected character after line continuation."
            return new_code, diff, explanation

        # NameError
        m = re.search(r"name '([^']+)' is not defined", stderr, re.IGNORECASE)
        if m:
            varname = m.group(1)
            default_val = "0"
            new_code = f"# Auto-added to fix NameError for '{varname}'\n{varname} = {default_val}\n" + code
            diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
            explanation = f"Defined variable '{varname}' at top with default {default_val}."
            return new_code, diff, explanation

        # IndexError or list index out of range
        if "indexerror" in stderr_low or "list index out of range" in stderr_low:
            lines = code.splitlines()
            target_i = None

            # Prefer line number from traceback when available
            file_line = parsed.get("file_line")
            if file_line:
                mline = re.search(r'line\s+(\d+)', file_line)
                if mline:
                    lineno = int(mline.group(1))
                    if 1 <= lineno <= len(lines):
                        candidate = lines[lineno - 1]
                        if re.search(r"\w+\s*\[.+\]", candidate):
                            target_i = lineno - 1

            # fallback: first matching indexing line
            if target_i is None:
                for i, ln in enumerate(lines):
                    if re.search(r"\w+\s*\[.+\]", ln):
                        target_i = i
                        break

            if target_i is not None:
                ln = lines[target_i]
                leading_ws_match = re.match(r"^(\s*)", ln)
                leading_ws = leading_ws_match.group(1) if leading_ws_match else ""
                stripped = ln[len(leading_ws):]

                # Try to detect a preceding for-loop using simple pattern
                prev_for = None
                for j in range(target_i - 1, -1, -1):
                    s = lines[j].strip()
                    if not s:
                        continue
                    # look for "for <var> in"
                    mfor = re.match(r"for\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+in\s+(.+):?", s)
                    if mfor:
                        loop_var = mfor.group(1)
                        prev_for = (j, loop_var)
                    break

                # If we found a for-loop and the index expression references the loop var,
                # try a guarded bounds-check (safe and preserves structure).
                if prev_for:
                    loop_var = prev_for[1]
                    if re.search(rf"\b{re.escape(loop_var)}\b", stripped):
                        # attempt to find array name inside the indexing expression; be defensive
                        arr_name = None
                        arr_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\[', stripped)
                        if arr_match:
                            arr_name = arr_match.group(1)
                        else:
                            # fallback: try to parse a simple expression before '['
                            try:
                                arr_name = stripped.split('[')[0].strip()
                                # sanitize arr_name to a simple token if possible
                                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', arr_name):
                                    arr_name = None
                            except Exception:
                                arr_name = None

                        # If we couldn't detect a valid array name, fall back to try/except
                        if arr_name:
                            guard_line = f"{leading_ws}if {loop_var} < len({arr_name}):"
                            inner_line = f"{leading_ws}    {stripped}"
                            new_lines = lines[:target_i] + [guard_line, inner_line] + lines[target_i + 1:]
                            new_code = "\n".join(new_lines)
                            diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
                            explanation = f"Added bounds-check `if {loop_var} < len({arr_name}):` around the indexing operation (heuristic)."
                            return new_code, diff, explanation
                        # else fall through to try/except wrapping

                # If no good bounds-check possible, wrap preserving indentation
                # Construct:
                # <leading_ws>try:
                # <leading_ws>    <stripped>
                # <leading_ws>except IndexError:
                # <leading_ws>    print('IndexError handled')
                wrapped = lines[:target_i] + [
                    f"{leading_ws}try:",
                    f"{leading_ws}    {stripped}",
                    f"{leading_ws}except IndexError:",
                    f"{leading_ws}    print('IndexError handled')"
                ] + lines[target_i + 1:]
                new_code = "\n".join(wrapped)
                diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
                explanation = "Wrapped indexing operation in try/except with preserved indentation to avoid IndexError (heuristic)."
                return new_code, diff, explanation

            # fallback wrap entire program
            new_code = "try:\n" + textwrap.indent(code, "    ") + "\nexcept IndexError:\n    print('IndexError handled')\n"
            diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
            return new_code, diff, "Wrapped entire program in try/except for IndexError (fallback)."

        # IndentationError (actual tab characters)
        if "indentationerror" in stderr_low:
            new_code = code.replace("\t", "    ")
            diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
            explanation = "Replaced tabs with 4 spaces to fix indentation."
            return new_code, diff, explanation

        # SyntaxError: missing colon heuristic
        if "syntaxerror" in stderr_low:
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
                explanation = "Added missing colon(s) in control/function lines (heuristic)."
                return new_code, diff, explanation

        # TypeError: concatenation between str and int (print)
        if "typeerror" in stderr_low and "can only concatenate" in stderr_low:
            new_code = re.sub(r'print\((["\'].*?["\'])\s*\+\s*([a-zA-Z_][a-zA-Z0-9_]*)\)', r'print(\1, \2)', code)
            if new_code != code:
                diff = "\n".join(difflib.unified_diff(code.splitlines(), new_code.splitlines(), lineterm=""))
                explanation = "Converted string concatenation in print to comma-separated arguments."
                return new_code, diff, explanation

        # ZeroDivisionError
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

        # No match
        return None, None, None

    except Exception as e:
        # defensive: print to server logs and return no patch (so the loop stops gracefully)
        print("Internal patch-generator error:", repr(e))
        return None, None, None


# ---------- Repair loop (Run -> Observe -> Patch -> Re-run) ----------
def repair_code_loop(initial_code: str, max_iterations: int = 3, timeout_seconds: int = 3):
    """
    Iteratively attempts to run code, generate patch when it fails, and re-run.
    Returns dictionary with final_code, patches (list), logs (iteration outputs), success flag.
    """
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

        # attempt to generate a patch
        new_code, diff, explanation = generate_patch_rule_based_v2(code, run_res["stderr"])
        if not new_code:
            # no applicable rule
            break

        patches.append({"iteration": it, "diff": diff, "explanation": explanation})
        code = new_code

    return {"final_code": code, "patches": patches, "logs": logs, "success": success}

# ---------- API endpoints ----------
@app.post("/run")
def run(req: RunRequest):
    if not req.code or not req.code.strip():
        raise HTTPException(status_code=400, detail="Empty code.")
    result = repair_code_loop(req.code, max_iterations=req.max_iterations, timeout_seconds=req.timeout_seconds)
    return result

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Serve a minimal static UI from / (optional) ----------
@app.get("/", include_in_schema=False)
def index_html():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return {"info": "Place static/index.html to use the demo UI."}

# ---------- Run if main ----------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
