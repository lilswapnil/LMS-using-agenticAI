#!/usr/bin/env python3
from __future__ import annotations
import json, os, math, argparse, random
from typing import Dict, Any, List

def fmt_num(x: float | int) -> str:
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)

def normalize_ascii(s: str) -> str:
    # Optional ASCII normalization for math symbols
    return (
        s.replace("≥", ">=")
         .replace("≤", "<=")
         .replace("±", "+/-")
         .replace("∞", "infinity")
         .replace("√", "sqrt")
    )

def write_jsonl(path: str, rows: List[Dict[str, Any]], ascii: bool) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            if ascii:
                # normalize all string fields recursively
                def norm(v):
                    if isinstance(v, str): return normalize_ascii(v)
                    if isinstance(v, list): return [norm(x) for x in v]
                    if isinstance(v, dict): return {k: norm(v2) for k, v2 in v.items()}
                    return v
                r = norm(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def gen_abs_eq(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    items=[]
    for _ in range(n):
        a = rng.randint(1,9)
        b = rng.randint(-10,10)
        x = rng.randint(-6,6)
        c = a*abs(x)+b
        instr = f"Solve for x: {a}|x| + {b} = {c}"
        steps=[
            {"thought":"Isolate |x| by subtracting b.","action":{"tool":"algebra","args":{"op":"subtract","value":b}},"observation":f"{a}|x| = {fmt_num(c-b)}"},
            {"thought":"Divide by a to get |x|.","action":{"tool":"algebra","args":{"op":"divide","value":a}},"observation":f"|x| = {fmt_num((c-b)/a)}"},
            {"thought":"Split into two cases.","action":{"tool":"algebra","args":{"op":"cases"}},"observation":f"x = ±{abs(x)}"}
        ]
        final=f"x = {x} or x = {-x}"
        items.append({"instruction":instr,"tools":[{"name":"algebra","desc":"abs equations","args_schema":{}}],"trajectory":steps,"final_answer":final})
    return items

def gen_radicals(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    items=[]
    ks = [4,8,12,18,20,24,27,32,45,50,72,98,63,75,96]
    for _ in range(n):
        k = rng.choice(ks)
        instr = f"Simplify: √{k}"
        a=1; b=k
        for t in [100,81,64,49,36,25,16,9,4]:
            if k%t==0:
                a=int(math.isqrt(t)); b=k//t; break
        steps=[{"thought":"Factor out the largest perfect square.","action":{"tool":"algebra","args":{"op":"factor_square"}},"observation":f"√{k} = {a}√{b}"}]
        final = f"{a}√{b}" if a!=1 else f"√{b}"
        items.append({"instruction":instr,"tools":[{"name":"algebra","desc":"radicals","args_schema":{}}],"trajectory":steps,"final_answer":final})
    return items

def gen_piecewise_eval(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    items=[]
    for _ in range(n):
        # f(x)= {2x+1, x<0; x^2, x>=0}
        x = rng.randint(-5,5)
        instr = f"Evaluate f(x) where f(x)=2x+1 for x<0 and f(x)=x^2 for x≥0 at x={x}."
        if x < 0:
            val = 2*x + 1
            obs = "Use 2x+1 since x<0."
        else:
            val = x*x
            obs = "Use x^2 since x≥0."
        steps=[
            {"thought":"Choose correct branch by x.","action":{"tool":"algebra","args":{"op":"branch","x":x}},"observation":obs},
            {"thought":"Compute the value.","action":{"tool":"algebra","args":{"op":"compute"}},"observation":f"f({x}) = {fmt_num(val)}"}
        ]
        items.append({"instruction":instr,"tools":[{"name":"algebra","desc":"piecewise evaluation","args_schema":{}}],"trajectory":steps,"final_answer":f"f({x}) = {fmt_num(val)}"})
    return items

def gen_functions_domain_range(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    items=[]
    for _ in range(n):
        a = rng.randint(1,4)
        b = rng.randint(1,5)
        instr = f"State the domain and range of f(x) = {a}x^2 + {b}."
        steps=[{"thought":"Quadratic opens up; vertex at y=b when a>0.","action":{"tool":"algebra","args":{"op":"analyze_quadratic"}}, "observation":f"Domain: all real numbers; Range: [ {b}, ∞ )"}]
        items.append({"instruction":instr,"tools":[{"name":"algebra","desc":"functions","args_schema":{}}],"trajectory":steps,"final_answer":f"Domain: (-∞, ∞); Range: [{b}, ∞)"})
    return items

def gen_exp_rules(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    items=[]
    for _ in range(n):
        p = rng.randint(1,5); q = rng.randint(1,5)
        instr = f"Simplify: (x^{p})/(x^{q})"
        e = p-q
        if e == 0:
            final = "1"
        elif e > 0:
            final = f"x^{e}"
        else:
            final = f"1/x^{abs(e)}"
        steps=[{"thought":"Subtract exponents when dividing same base.","action":{"tool":"algebra","args":{"rule":"a^m/a^n=a^(m-n)"}}, "observation":final}]
        items.append({"instruction":instr,"tools":[{"name":"algebra","desc":"exponent division","args_schema":{}}],"trajectory":steps,"final_answer":final})
    return items

def gen_linear_word(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    items=[]
    for _ in range(n):
        p = rng.randint(5,20); q = rng.randint(1,10)
        instr = f"A gym charges a ${p} signup fee and ${q} per month. Write and evaluate the cost after 6 months."
        cost = p + q*6
        steps=[{"thought":"Cost = signup + monthly*months.","action":{"tool":"algebra","args":{"op":"model"}}, "observation":f"C= {p} + {q}*6 = {cost}"}]
        items.append({"instruction":instr,"tools":[{"name":"algebra","desc":"linear model","args_schema":{}}],"trajectory":steps,"final_answer":f"C = ${cost}"} )
    return items

def gen_quadratic_word(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    items=[]
    for _ in range(n):
        v = rng.randint(-4,4)
        a = rng.randint(1,3)
        instr = f"A projectile’s height is h(t) = -{a}t^2 + {2*a}t + {v}. Find the vertex (time of max height)."
        t = 1  # for -a t^2 + (2a) t + v, vertex t = b/(2a) = 1
        h = -a*(t**2) + 2*a*t + v  # equals a + v
        steps=[{"thought":"Vertex at t=-b/(2a). For -at^2+bt+c, t=b/(2a).","action":{"tool":"algebra","args":{"op":"vertex"}}, "observation":f"t={t}, h={fmt_num(h)}"}]
        items.append({"instruction":instr,"tools":[{"name":"algebra","desc":"quadratic modeling","args_schema":{}}],"trajectory":steps,"final_answer":f"Vertex at t={t}, height={fmt_num(h)}"} )
    return items

def build(out_sft: str, out_eval: str, ascii: bool, seed: int,
          n_abs: int, n_rad: int, n_piece: int, n_fdr: int, n_exp: int, n_lin: int, n_quad: int,
          eval_size: int) -> None:
    rng = random.Random(seed)
    data: List[Dict[str, Any]] = []
    data += gen_abs_eq(n_abs, rng)
    data += gen_radicals(n_rad, rng)
    data += gen_piecewise_eval(n_piece, rng)
    data += gen_functions_domain_range(n_fdr, rng)
    data += gen_exp_rules(n_exp, rng)
    data += gen_linear_word(n_lin, rng)
    data += gen_quadratic_word(n_quad, rng)
    rng.shuffle(data)

    write_jsonl(out_sft, data, ascii)
    eval_rows = [{"query": x["instruction"], "expected": x["final_answer"]} for x in data[:eval_size]]
    write_jsonl(out_eval, eval_rows, ascii)

    print({"sft": len(data), "eval": len(eval_rows), "out_sft": out_sft, "out_eval": out_eval})

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate algebra SFT and eval JSONL.")
    ap.add_argument("--out-sft", default="data/algebra_plus_sft.jsonl")
    ap.add_argument("--out-eval", default="data/algebra_plus_eval.jsonl")
    ap.add_argument("--ascii", action="store_true", help="Normalize math symbols to ASCII")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-abs", type=int, default=60)
    ap.add_argument("--n-rad", type=int, default=60)
    ap.add_argument("--n-piece", type=int, default=50)
    ap.add_argument("--n-fdr", type=int, default=50)
    ap.add_argument("--n-exp", type=int, default=50)
    ap.add_argument("--n-lin", type=int, default=50)
    ap.add_argument("--n-quad", type=int, default=40)
    ap.add_argument("--eval-size", type=int, default=200)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build(
        out_sft=args.out_sft, out_eval=args.out_eval, ascii=args.ascii, seed=args.seed,
        n_abs=args.n_abs, n_rad=args.n_rad, n_piece=args.n_piece, n_fdr=args.n_fdr,
        n_exp=args.n_exp, n_lin=args.n_lin, n_quad=args.n_quad, eval_size=args.eval_size
    )
