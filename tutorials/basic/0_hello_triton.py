import triton
import triton.language as tl

@triton.jit
def hello_triton():
    tl.device_print("Hello Triton!")

if __name__ == "__main__":
    hello_triton[(1,)]() 
    # grid를 통해, 내가 사용하고자 하는 SM을 지정해줄 수 있다.
    # grid는 3차원 tuple이다. 아무래도 hardware의 코어의 위치와 관련이 있는 것 같다.
    # x, y, z축으로 몇 개의 코어를 선택할 것인가? -> 그 위에서 커널 함수를 돌린다.