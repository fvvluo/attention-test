"""B5 warp-MMA GQA decode package (independent implementation by Liu Xiaochen).

Design reference (idea only, no code copied): M16 Pack-GQA + LSE-weighted
split combine, learned from studying quanbofeng's decode. Structure/layouts/
softmax/combine written from scratch with public CUTLASS CuTe DSL.
"""
