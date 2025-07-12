from __future__ import annotations

import json

from expression_distance import master_convert, sympy_to_tree
from latex_processor import *  # noqa: F403, F401

# 读取 Excel 文件（默认读取第一个工作表）
file_path = "./solutions/claude-sonnet-4-0514.json"
with open(file_path, encoding="utf-8") as file:
    data = json.load(file)


def write_file(s: str) -> None:
    with open("validator_logs.txt", "a", encoding="utf-8") as f:
        f.write(s)


with open("validator_logs.txt", "w") as f:
    f.write("")

formulas: list[str] = []
for i in data:
    formulas.append(i["answer"])
# print(formulas)
cnt = 0

formulas = formulas[0:1000]
opt: list[dict[str, str]] = []
for formula_a in formulas:
    formula = formula_a
    cnt += 1
    # master_convert(formula)
    try:
        expre = master_convert(formula)
        tree = sympy_to_tree(expre)
        print(f"Expression{cnt}/{len(formulas)}Validated")
    except Exception:
        print(f"Wrong formatted Expression{formula}")
        write_file(str(formula))
        write_file("\n")
        opt.append({"answer": formula})

with open("wrong_expressions.json", "w", encoding="utf-8") as f:
    json.dump(opt, f, ensure_ascii=False, indent=4)
print("validation complete!")
