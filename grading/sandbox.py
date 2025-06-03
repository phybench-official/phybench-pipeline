
from latex2sympy2_extended import latex2sympy


from sympy import posify
import json

from sympy import symbols, sympify

from latex_pre_process import *



s= "\\dfrac{u\\sqrt{u^{2} + 2gH}{g}}"
print(master_convert(s))
