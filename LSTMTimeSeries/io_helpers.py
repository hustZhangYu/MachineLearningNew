"""CSV vector parsing without training or file reads at import time."""
import ast


def str_to_list(s):
    return ast.literal_eval(s)
