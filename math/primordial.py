"""
Basic Usage of Primary Mathematics
"""
import math
import sympy
from sympy.abc import *


alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# Basic Usages
def C(n,m):
    # Cnm means the combinational number of n boxes with m balls
    return math.factorial(n)//(math.factorial(m)*math.factorial(n-m))

# Check if the number input is a prime or not
def is_prime(n):
    flag = True
    if n > 0:
        if n != 2:
            for k in range(2,n):
                if n%k == 0:
                    flag = False
        else:
            return False
    else:
        return False
    return flag

def expand_symbol(expr):
    expr = regular_expression(expr)
    return sympy.expand(expr)

def solve_eq(expr):

    try:
        return sympy.solve(regular_expression(expr))
    except:
        return sympy.solve([regular_expression(expr[0]),regular_expression(expr[1])])

def factor_symbol(expr):
    expr = regular_expression(expr)
    return sympy.factor(expr)

def collect_symbol(expr,var):
    var = symbolize(var)
    expr = regular_expression(expr)
    return sympy.collect(expr,var)

def subs_symbol(expr,symb,n):
    return expr.subs(symb,n)

def symbolize(expr):
    return sympy.sympify(expr)

def simplify(expr):
    expr = regular_expression(expr)
    return sympy.simplify(expr)

def cancel_fact(expr):
    return sympy.cancel(expr)

def diffi(expr,var,n):
    var = symbolize(var)
    n = int(n)
    expr = symbolize(regular_expression(expr))
    return expr.diff(var,n)

def num_lf(symb,n):
    return symb.evalf(n)

def GCD(a,b):
    a = regular_expression(a)
    b = regular_expression(b)
    a = symbolize(a)
    b = symbolize(b)
    return sympy.gcd(a,b)

def LCM(a,b):
    a = regular_expression(a)
    b = regular_expression(b)
    a = symbolize(a)
    b = symbolize(b)
    return sympy.lcm(a,b)
    
def syns(a,b):
    return [a,b]

def str_to_list(s):
    rt = []
    for e in s:
        rt.append(e)
    return rt

def is_func(expr):
    func = expr[0:4]
    if func[0] in alphabet and func[2] in alphabet: 
        if func[1] == "(" and func[3] == ")":
            return True
        else:
            return False
    else:
        return False

def regular_expression(expr):
    expr = str(expr)
    if "=" in expr:
        ind = str_to_list(expr).index("=")

        flag = is_func(expr)
        if flag:
            fun_name = expr[0:str_to_list(expr).index("(")]
            rst = expr[ind+1:len(str_to_list(expr))]


            return fun_name + " ="+ rst

        else:
            fst = expr[0:ind]
            rst = expr[ind+1:len(str_to_list(expr))]
            return sympy.Eq(symbolize(fst),symbolize(rst))

    else:
        return str(expr)

regular_expression("f(x) = x")
def find_sol(Set,symb):
    for i in range(len(Set)):
        flag = eval("Set[{}] == symb".format(i))
        if flag:
            return Set[i +1]

def Add(a,b):
    return symbolize(regular_expression(a)) + symbolize(regular_expression(b))

def Substract(a,b):
    return symbolize(regular_expression(a)) - symbolize(regular_expression(b))

def Muti(a,b):
    a = symbolize(a)
    b = symbolize(b)
    return a * b

def Divi(a,b):
    a = symbolize(a)
    b = symbolize(b)
    return a / b

def Diff(expr,var):
    var = symbolize(var)
    expr = regular_expression(expr)
    expr = symbolize(expr)
    return expr.diff(var)

def Sort_Ascend(L):
    A = list(L)
    A.sort()
    return L

def Sort_Descend(L):
    A = list(L)
    A.sort()
    A.reverse()
    return L