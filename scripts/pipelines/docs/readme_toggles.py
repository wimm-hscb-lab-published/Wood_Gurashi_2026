"""Make the package READMEs easier to scan, and keep their links to the code up to date.

    python docs/readme_toggles.py convert DNA_Utils CNV_Utils LOH_Pipeline PRO_Utils
    python docs/readme_toggles.py refresh DNA_Utils CNV_Utils LOH_Pipeline PRO_Utils

convert  Turns each function section (a "### name(...)" heading inside a module section) into a
         collapsible block: the summary line shows the name and the first sentence; the block holds
         the full signature, a link to the definition line in the code, and the original text.
         "Private helpers" tables are collapsed too, with each helper linked to its line.
         Run once; running it again leaves already-converted READMEs unchanged.
refresh  Updates the line numbers of every code link after the code has been edited.
         Links look like [module.py:123](module.py#L123 "name"); the title holds the name to look up.

Run from the pipelines folder.
"""
from __future__ import annotations

import ast
import html
import os
import re
import sys

LINK = re.compile(r'\[([^\]]*)\]\(([\w/.]+\.py)#L(\d+) "([\w.]+)"\)')
HEADING = re.compile(r"^(#{2,4}) (.+?)\s*$")
FENCE = re.compile(r"^\s*```")
HELPER_TABLES = {"private helpers", "label helpers", "other helpers", "constants"}


def definitions(path: str) -> dict[str, int]:
    """Module-level functions, classes and assigned names → line number (last definition wins)."""
    tree = ast.parse(open(path).read())
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = node.lineno
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out[f"{node.name}.{sub.name}"] = sub.lineno
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node.lineno
    return out


def link(module_file: str, name: str, line: int, text: str | None = None) -> str:
    return f'[{text or f"{module_file}:{line}"}]({module_file}#L{line} "{name}")'


def first_sentence(body: list[str]) -> str:
    in_fence = False
    for raw in body:
        line = raw.strip()
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or not line or line.startswith(("!", "|", "<", "#", "-", "*", ">", "1.")):
            continue
        sentence = re.split(r"(?<=[.!?])\s+(?=[A-Z`(])", line)[0]
        return sentence if len(sentence) <= 160 else sentence[:157].rstrip() + "…"
    return ""


def md_inline_to_html(text: str) -> str:
    """Escape for use inside <summary>, keeping `code` and **bold**."""
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append(f"<code>{html.escape(p[1:-1])}</code>")
        else:
            p = html.escape(p)
            out.append(re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", p))
    return "".join(out)


def heading_names(title: str) -> list[str]:
    """Names documented by a heading such as 'f(a, b)', 'f / g', 'A, B, C' or 'Cls (dataclass)'."""
    head = re.split(r"[(\s]", title.strip(), maxsplit=1)[0].rstrip(",")
    if not re.match(r"^[A-Za-z_][\w.]*$", head):
        return []
    names = [head]
    for extra in re.findall(r"(?:/|,)\s*([A-Za-z_]\w*)", re.sub(r"\(.*?\)", "", title)):
        if extra not in names:
            names.append(extra)
    return names


def convert(pkg: str) -> int:
    path = os.path.join(pkg, "README.md")
    lines = open(path).read().split("\n")
    modules = {os.path.splitext(f)[0] for f in os.listdir(pkg) if f.endswith(".py")}
    defs_cache: dict[str, dict[str, int]] = {}

    out, i, module, n_conv, in_fence = [], 0, None, 0, False
    while i < len(lines):
        line = lines[i]
        if FENCE.match(line):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        m = HEADING.match(line) if not in_fence else None
        if not m:
            out.append(line)
            i += 1
            continue
        level, title = len(m.group(1)), m.group(2)
        if level == 2:
            module = title.strip() if title.strip() in modules else None
            out.append(line)
            i += 1
            continue
        if module is None:
            out.append(line)
            i += 1
            continue

        # body = until next heading of the same or higher level (outside fences)
        j, fence = i + 1, False
        while j < len(lines):
            if FENCE.match(lines[j]):
                fence = not fence
            elif not fence:
                hm = HEADING.match(lines[j])
                if hm and len(hm.group(1)) <= level:
                    break
            j += 1
        body = lines[i + 1:j]
        while body and body[-1].strip() in ("", "---"):
            body.pop()
        trailing = lines[i + 1 + len(body):j]

        module_file = f"{module}.py"
        defs = defs_cache.setdefault(module, definitions(os.path.join(pkg, module_file)))
        names = [n for n in heading_names(title) if n in defs]

        if title.strip().lower() in HELPER_TABLES:
            new_body = []
            for row in body:
                if row.startswith("|"):
                    row = re.sub(r"^\|\s*`([\w.]+)`",
                                 lambda mm: f"| {link(module_file, mm.group(1), defs[mm.group(1)], f'`{mm.group(1)}`')}"
                                 if mm.group(1) in defs else mm.group(0), row)
                new_body.append(row)
            n_rows = sum(1 for r in body if r.startswith("|")) - 2
            label = title.strip()
            out += ["<details>", f"<summary><b>{html.escape(label)}</b> ({n_rows} entries)</summary>", ""]
            out += new_body + ["", "</details>"] + trailing
            i = j
            n_conv += 1
            continue

        if not names:
            out.append(line)
            i += 1
            continue

        shown = " / ".join(names)
        summary = first_sentence(body)
        links = " · ".join(link(module_file, n, defs[n]) for n in names)
        out += [
            "<details>",
            f"<summary><b><code>{html.escape(shown)}</code></b>"
            + (f" — {md_inline_to_html(summary)}" if summary else "") + "</summary>",
            "",
            f"`{title.strip()}`",
            "",
            f"Source: {links}",
            "",
        ]
        out += body + ["", "</details>", ""] + [t for t in trailing if t.strip()]
        if trailing and not trailing[-1].strip():
            out.append("")
        i = j
        n_conv += 1

    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out))
    open(path, "w").write(text)
    return n_conv


def refresh(pkg: str) -> int:
    path = os.path.join(pkg, "README.md")
    text = open(path).read()
    cache: dict[str, dict[str, int]] = {}
    changed = 0

    def repl(m):
        nonlocal changed
        disp, file, line, name = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        defs = cache.setdefault(file, definitions(os.path.join(pkg, file)))
        if name not in defs:
            print(f"  WARNING {pkg}: {name} not found in {file}")
            return m.group(0)
        new = defs[name]
        if new != line:
            changed += 1
        disp = re.sub(rf"^{re.escape(file)}:\d+$", f"{file}:{new}", disp)
        return f'[{disp}]({file}#L{new} "{name}")'

    open(path, "w").write(LINK.sub(repl, text))
    return changed


if __name__ == "__main__":
    mode, pkgs = sys.argv[1], sys.argv[2:]
    for pkg in pkgs:
        if mode == "convert":
            if "<details>" in open(os.path.join(pkg, "README.md")).read():
                print(f"{pkg}: already converted, refreshing links instead")
                print(f"{pkg}: {refresh(pkg)} line numbers updated")
            else:
                print(f"{pkg}: {convert(pkg)} sections collapsed")
        elif mode == "refresh":
            print(f"{pkg}: {refresh(pkg)} line numbers updated")
        else:
            sys.exit(__doc__)
