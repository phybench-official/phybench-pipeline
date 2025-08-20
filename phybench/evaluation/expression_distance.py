from __future__ import annotations

import threading
from collections.abc import Callable
from enum import Enum
from time import perf_counter
from typing import Any, Final, TypeVar, overload

import sympy
from loguru import logger
from sympy import (
    Add,
    Float,
    Function,
    Integer,
    Mul,
    Pow,
    Rational,
    Symbol,
    count_ops,
    expand,
    posify,
    simplify,
)
from sympy import (
    E as EulerNumber,
)
from sympy import (
    oo as Infinity,
)
from sympy import (
    pi as Pi,
)

from phybench.settings import EvaluationEEDSettings

from .latex_processor import master_convert
from .tree_distance import ext_distance

F = TypeVar("F", bound=Callable[..., Any])


class NodeType(str, Enum):
    NUMBER = "number"
    SYMBOL = "symbol"
    OPERATOR = "operator"
    FUNCTION = "function"


NegativeInfinity: Final[sympy.Basic] = -sympy.oo

# --- Minimal problem-id context for logging ---
_CURRENT_PROBLEM_ID: int | None = None
_CURRENT_TREE_SIDE: str | None = None


def set_problem_context(problem_id: int) -> None:
    """Set current problem id for logging (best-effort, no behavior change)."""
    global _CURRENT_PROBLEM_ID
    _CURRENT_PROBLEM_ID = problem_id


def clear_problem_context() -> None:
    """Clear current problem id context."""
    global _CURRENT_PROBLEM_ID
    _CURRENT_PROBLEM_ID = None
    clear_tree_side()


def set_tree_side(side: str) -> None:
    """Set which side is currently being transformed into a tree (for logging)."""
    global _CURRENT_TREE_SIDE
    _CURRENT_TREE_SIDE = side


def clear_tree_side() -> None:
    global _CURRENT_TREE_SIDE
    _CURRENT_TREE_SIDE = None


def _log_latex_convert_error(
    side: str, exc: Exception, content: str, known: bool
) -> None:
    """Log a concise LaTeX conversion error entry.

    Args:
        side: "GT" or "GEN"
        exc: caught exception
        content: the LaTeX text that failed to convert
        known: True for known parsing errors (SyntaxError/ValueError/TypeError), else False
    """
    pid = _CURRENT_PROBLEM_ID if _CURRENT_PROBLEM_ID is not None else "unknown"
    level = logger.warning if known else logger.error
    kind = "known parsing error" if known else "UNEXPECTED error"
    level(
        f"Problem ID {pid}: LaTeX convert failed ({side}) - {kind}: {type(exc).__name__}: {exc} - {side}='{content}'"
    )


"""
Guide:
You only need to use EED and install the following packages:
- sympy
- numpy
- latex2sympy2_extended
"""

"""
There are four main categories:

Constants: integers, decimals, or mathematical constants like π and e.
Variables: letters like x, y, z, or specified terms in problems (e.g., ħ, c, G).
Functions: sine, cosine, exponential, logarithm, etc.
Operators: basic binary operations including addition, multiplication, and exponentiation.
"""


def get_node_type(node: TreeNode) -> str:
    """Extracts the base type from a node's label."""
    return node.label.split("_")[0]


def update_func(x: TreeNode, y: TreeNode, eed_settings: EvaluationEEDSettings) -> float:
    if x.label == y.label:
        return 0

    x_type = get_node_type(x)
    y_type = get_node_type(y)

    if x_type == y_type:
        return eed_settings.update_cost[x_type]
    return eed_settings.change_type_cost


def remove_func(x: TreeNode, eed_settings: EvaluationEEDSettings) -> float:
    return eed_settings.delete_cost[get_node_type(x)]


def remove_tree_func(x: TreeNode, eed_settings: EvaluationEEDSettings) -> float:
    if not x.children:
        return remove_func(x, eed_settings)
    s = calc_tree_size(x, eed_settings)
    return min(
        s,
        eed_settings.discount_slope * (s - eed_settings.bar_size)
        + eed_settings.bar_size,
    )


def insert_func(x: TreeNode, eed_settings: EvaluationEEDSettings) -> float:
    return eed_settings.insert_cost[get_node_type(x)]


def insert_tree_func(x: TreeNode, eed_settings: EvaluationEEDSettings) -> float:
    return remove_tree_func(x, eed_settings)


def calc_tree_size(node: TreeNode, eed_settings: EvaluationEEDSettings) -> float:
    """
    Calculate the size of a subtree based on its total insertion cost.
    The function computes the size of a subtree by summing up the insertion
    costs of the current node and all its descendant nodes. If the subtree
    size has already been calculated and stored in `node.subtree_size`, it
    returns the cached value to avoid redundant computation.
    Args:
        node (Node): The root node of the subtree for which the size is to
                     be calculated
    Returns:
        int: The total size of the subtree, calculated as the sum of the
             insertion costs of the current node and all its descendants.
    Notes:
        - The `insert_cost` dictionary is assumed to be globally defined
          and maps node labels to their respective insertion costs.
        - The function modifies the `subtree_size` attribute of the input
          node to store the calculated subtree size for future use.
    """

    total = eed_settings.insert_cost[get_node_type(node)]

    if node.children and node.subtree_size != 0:
        return node.subtree_size

    for child in node.children:
        total += calc_tree_size(child, eed_settings)

    node.subtree_size = total

    return total


"""
Scoring function from relative distance
"""


def score_calc(
    tree_dist: float, tree_size: float, eed_settings: EvaluationEEDSettings
) -> float:
    if tree_dist == 0.0:
        return 100
    return max(
        0,
        eed_settings.initial_score - eed_settings.scoring_slope * tree_dist / tree_size,
    )


class TimeoutError(Exception):
    pass


def with_timeout(timeout_seconds: float) -> Callable[[F], F]:
    """Windows-compatible timeout decorator using threading"""

    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result: list[Any] = []
            exception: list[Exception] = []

            def target() -> None:
                try:
                    result.append(func(*args, **kwargs))
                except Exception as e:
                    exception.append(e)

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)

            if thread.is_alive():
                raise TimeoutError(
                    f"Function timed out after {timeout_seconds} seconds"
                )

            if exception:
                raise exception[0]

            return result[0] if result else None

        return wrapper  # type: ignore

    return decorator


def simplify_with_timeout(expr: Any, timeout: float) -> Any:
    @with_timeout(timeout)
    def _simplify(expr: Any) -> Any:
        return simplify(expr)

    return _simplify(expr)


def time_simplify(
    expr: Any,
    timeout: float,
    *,
    side: str | None = None,
    strategy: str = "sympy.simplify",
) -> Any:
    start = perf_counter()
    try:
        result = simplify_with_timeout(expr, timeout)
        return result
    except TimeoutError:
        elapsed = perf_counter() - start
        pid = _CURRENT_PROBLEM_ID if _CURRENT_PROBLEM_ID is not None else "unknown"
        size_chars = len(str(expr))
        try:
            ops = int(count_ops(expr))
        except Exception:
            ops = -1
        which = side if side is not None else "unknown"
        logger.warning(
            f"Problem ID {pid}: Simplification timed out ({which}). Elapsed={elapsed:.2f}s, threshold={timeout:.2f}s, size={size_chars} chars, ops={ops}. Strategy={strategy}. Fallback=return-original"
        )
        return expr


def equal_with_timeout(expr1: Any, expr2: Any, timeout: float) -> bool:
    @with_timeout(timeout)
    def _equals(expr1: Any, expr2: Any) -> bool:
        return bool(expr1.equals(expr2))

    result = _equals(expr1, expr2)
    return bool(result) if result is not None else False


def time_equal(expr1: Any, expr2: Any, timeout: float) -> bool:
    try:
        result = equal_with_timeout(expr1, expr2, timeout)
        return result
    except TimeoutError:
        return False


def sympy_to_tree(expr: Any) -> TreeNode:
    """
    Convert a SymPy expression into a tree structure.
    This function takes a SymPy expression and recursively converts it into a tree
    representation using `TreeNode` objects. Each node in the tree is labeled based
    on the type of the SymPy expression (e.g., number, symbol, operator, or function),
    and its children represent the arguments of the expression.
    Args:
        expr (sympy.Basic): The SymPy expression to be converted.
    Returns:
        TreeNode: The root node of the tree representation of the SymPy expression.
    Raises:
        ValueError: If the SymPy expression contains an unsupported type.
    Supported Types:
        - Numbers: Integer, Pi, EulerNumber, Float, Rational, Infinity, NegativeInfinity
        - Symbols: Symbol
        - Binary Operators: Add, Mul, Pow
        - Functions: Any subclass of `sympy.Function`
    Example:
        >>> from sympy import symbols, sin, pi
        >>> x, y = symbols('x y')
        >>> expr = x + y * sin(pi)
        >>> tree = sympy_to_tree(expr)
        >>> print(tree)
    """
    # Symbols and constants
    if isinstance(expr, Integer | Float | Rational) or expr in (
        Pi,
        EulerNumber,
        Infinity,
        NegativeInfinity,
    ):
        return TreeNode(label=f"{NodeType.NUMBER.value}_{expr}", children=[])
    elif isinstance(expr, Symbol):
        return TreeNode(label=f"{NodeType.SYMBOL.value}_{expr}", children=[])

    # Binary operators
    elif isinstance(expr, Add | Mul | Pow):
        op_name = type(expr).__name__
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label=f"{NodeType.OPERATOR.value}_{op_name}", children=children)

    # Functions
    elif isinstance(expr, Function):
        func_name = expr.func.__name__
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(
            label=f"{NodeType.FUNCTION.value}_{func_name}", children=children
        )

    else:
        pid = _CURRENT_PROBLEM_ID if _CURRENT_PROBLEM_ID is not None else "unknown"
        side = _CURRENT_TREE_SIDE if _CURRENT_TREE_SIDE is not None else "unknown"
        err_code = (
            "unsupported_type_imaginary_unit"
            if type(expr).__name__ == "ImaginaryUnit"
            else "unsupported_sympy_type"
        )
        logger.error(
            f"Problem ID {pid}: Tree build failed ({side}) - {err_code}: Unsupported SymPy type: {type(expr).__name__}, subexpr={expr}"
        )
        raise ValueError(f"Unsupported SymPy type: {type(expr)}")


class TreeNode:
    @overload
    def __init__(self, label: str) -> None: ...

    @overload
    def __init__(self, label: str, children: list[TreeNode]) -> None: ...

    @overload
    def __init__(
        self, label: str, children: list[TreeNode], node_type: str
    ) -> None: ...

    def __init__(
        self,
        label: str,
        children: list[TreeNode] | None = None,
        node_type: str = "other",
    ) -> None:
        self.label = label
        self.children = children if children is not None else []
        self.node_type = node_type
        self.subtree_size: float = 0.0

    def get_children(self) -> list[TreeNode]:
        return self.children

    def __str__(self) -> str:
        return self.label


def print_tree(node: TreeNode, indent: int = 0) -> None:
    """Print a tree structure"""
    logger.debug("  " * indent + f"└─ {node.label}")
    for child in node.children:
        print_tree(child, indent + 1)


class LaTeXError(Exception):
    def __init__(self, message: str = "LaTeXError") -> None:
        super().__init__(message)


class SymPyError(Exception):
    def __init__(self, message: str = "SymPyError") -> None:
        super().__init__(message)


class TreeError(Exception):
    def __init__(self, message: str = "TreeError") -> None:
        super().__init__(message)


class DistError(Exception):
    def __init__(self, message: str = "DistanceError") -> None:
        super().__init__(message)


def EED(
    answer_latex: str,
    test_latex: str,
    eed_settings: EvaluationEEDSettings,
    debug_mode: bool = False,
) -> tuple[float, float, float, float]:
    """
    Computes the similarity score and distance metrics between two LaTeX expressions.
    This function evaluates the equivalence of two mathematical expressions represented
    in LaTeX format. It uses symbolic computation and tree-based distance metrics to
    calculate a similarity score and other related metrics.

        tuple: A tuple containing the following elements:
            - score (float): The similarity score between the two expressions (0 to 100).
            - relative_distance (float): The normalized distance between the two expressions.
            - answer_tree_size (int): The size of the expression tree for the answer.
            - distance (float): The raw distance between the two expression trees.
    Notes:
        - If either input contains unsupported LaTeX constructs (e.g., integrals or sums),
          the function returns default values indicating failure.
        - If the test expression is significantly longer than the answer expression,
          the function assumes they are not equivalent.
        - The function uses symbolic simplification and tree-based distance metrics to
          evaluate equivalence.
        - In case of errors during processing, the function returns default values unless
          `debug_mode` is enabled, in which case it raises specific exceptions.
    Exceptions:
        - LaTeXError: Raised when LaTeX conversion to symbolic expressions fails (if `debug_mode` is True).
        - SymPyError: Raised when symbolic simplification or tree construction fails (if `debug_mode` is True).
        - DistError: Raised when distance calculation fails (if `debug_mode` is True).
    Args:
        answer_latex: the latex expression of answer expression
        test_latex: the latex expression of test expression
        debug_mode: whether it raise errors or just skip it
    Returns:
         tuple: A tuple containing the following elements:
            - score (float): The similarity score between the two expressions (0 to 100).
            - relative_distance (float): The normalized distance between the two expressions.
            - answer_tree_size (int): The size of the expression tree for the answer.
            - distance (float): The raw distance between the two expression trees.
    """

    if not test_latex:
        return 0, -1, -1, -1
    if "\\int" in test_latex or "\\int" in answer_latex:
        return 0, -1, -1, -1
    if "\\sum" in test_latex or "\\sum" in answer_latex:
        return 0, -1, -1, 1
    if answer_latex == test_latex:
        return 100, 0.0, -1, 0
    if len(test_latex) > 3 * len(answer_latex):
        return 0, -1, -1, -1

    try:
        answer_exp = master_convert(answer_latex)
    except (SyntaxError, ValueError, TypeError) as e:
        _log_latex_convert_error("GT", e, answer_latex, known=True)
        if debug_mode:
            raise LaTeXError(f"Fail to convert latex (GT).\n GT:{answer_latex}") from e
        return 0, -1, -1, -1
    except Exception as e:
        _log_latex_convert_error("GT", e, answer_latex, known=False)
        if debug_mode:
            raise LaTeXError(
                "An unexpected error occurred during latex conversion (GT)."
            ) from e
        return 0, -1, -1, -1

    try:
        test_exp = master_convert(test_latex)
    except (SyntaxError, ValueError, TypeError) as e:
        _log_latex_convert_error("model answer", e, test_latex, known=True)
        if debug_mode:
            raise LaTeXError(
                f"Fail to convert latex (model answer).\n model answer:{test_latex}"
            ) from e
        return 0, -1, -1, -1
    except Exception as e:
        _log_latex_convert_error("model answer", e, test_latex, known=False)
        if debug_mode:
            raise LaTeXError(
                "An unexpected error occurred during latex conversion (model answer)."
            ) from e
        return 0, -1, -1, -1

    try:
        answer_exp, rep1 = posify(answer_exp)
        answer_exp = time_simplify(
            answer_exp, timeout=eed_settings.simplify_time_limit, side="GT"
        )

        test_exp, rep2 = posify(test_exp)
        test_exp = time_simplify(
            test_exp, timeout=eed_settings.simplify_time_limit, side="model answer"
        )

        answer_exp = answer_exp.subs(rep1)
        test_exp = test_exp.subs(rep2)

        zero_exp = time_simplify(
            expand(answer_exp - test_exp),
            timeout=eed_settings.simplify_time_limit,
            side="combined",
        )

        if answer_exp == test_exp or zero_exp == 0:
            return 100, 0.0, 0, 0

        if time_equal(
            answer_exp, test_exp, timeout=eed_settings.equals_time_limit
        ):  # equality check with a shorter timeout
            return 100, 0.0, 0, 0

    except (AttributeError, TypeError, ValueError) as e:
        logger.warning(
            f"Error during expression simplification, returning zero score: {type(e).__name__}: {e} - GT='{answer_latex}', model answer='{test_latex}'"
        )
        if debug_mode:
            raise SymPyError(
                f"Failed to simplify the sympy expression. Error: {e}"
            ) from e
        return 0, -1, -1, -1
    except Exception as e:
        logger.error(
            f"An UNEXPECTED error occurred during expression simplification: {type(e).__name__}: {e} - GT='{answer_latex}', model answer='{test_latex}'"
        )
        if debug_mode:
            raise SymPyError(
                "An unexpected error occurred during expression simplification."
            ) from e
        return 0, -1, -1, -1

    try:
        set_tree_side("GT")
        tree_answer = sympy_to_tree(answer_exp)
        set_tree_side("model answer")
        tree_test = sympy_to_tree(test_exp)
    except ValueError as e:
        logger.warning(
            f"Failed to build expression tree, returning zero score: {e} - GT='{answer_latex}', model answer='{test_latex}'"
        )
        if debug_mode:
            raise SymPyError(
                f"Failed to build the sympy expression tree.\nGT:{answer_exp}\nmodel answer:{test_exp}"
            ) from e
        return 0, -1, -1, -1
    except Exception as e:
        logger.error(
            f"An UNEXPECTED error occurred during tree construction: {type(e).__name__}: {e} - GT='{answer_latex}', model answer='{test_latex}'"
        )
        if debug_mode:
            raise SymPyError(
                "An unexpected error occurred during tree construction."
            ) from e
        return 0, -1, -1, -1
    finally:
        clear_tree_side()

    try:
        distance = ext_distance(
            tree_test,
            tree_answer,
            get_children=lambda x: x.get_children(),
            single_insert_cost=lambda x: insert_func(x, eed_settings),
            insert_cost=lambda x: insert_tree_func(x, eed_settings),
            single_remove_cost=lambda x: remove_func(x, eed_settings),
            remove_cost=lambda x: remove_tree_func(x, eed_settings),
            update_cost=lambda x, y: update_func(x, y, eed_settings),
        )
    except RecursionError as e:
        logger.warning(
            f"Failed to calculate distance due to recursion depth: {type(e).__name__}: {e}"
        )
        if debug_mode:
            raise DistError(
                f"Failed to calculate the distance between trees.\nGT:{answer_latex}\n GEN:{test_latex}"
            ) from e
        return 0, -1, calc_tree_size(tree_answer, eed_settings), -1
    except Exception as e:
        logger.error(
            f"An UNEXPECTED error occurred during distance calculation: {type(e).__name__}: {e}"
        )
        if debug_mode:
            raise DistError(
                "An unexpected error occurred during distance calculation."
            ) from e
        return 0, -1, calc_tree_size(tree_answer, eed_settings), -1
    tree_size = calc_tree_size(tree_answer, eed_settings)
    distance_number = distance

    relative_distance = distance / tree_size

    score = score_calc(distance_number, tree_size, eed_settings)

    return score, relative_distance, tree_size, distance_number
