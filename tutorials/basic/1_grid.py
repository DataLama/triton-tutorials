from typing import Dict

import triton
import triton.language as tl

@triton.jit
def print_grid():
    x_pid = tl.program_id(0) # x축의 프로세스 아이디
    y_pid = tl.program_id(1) # y축의 프로세스 아이디
    z_pid = tl.program_id(2) # z축의 프로세스 아이디
    tl.device_print("x_pid: ", x_pid)
    tl.device_print("y_pid: ", y_pid)
    tl.device_print("z_pid: ", z_pid)

def grid(meta:Dict):
    """
    Args: meta는 그리드를 결정하는데 사용할 수 있는 meta 정보
    """
    return (4, 2)

print_grid[grid]()
# python 1_grid.py | sort | uniq