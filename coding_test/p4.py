'''
스택은 먼저 넣은 자료가 나중에 나오는 형태의 자료구조입니다. 
자료를 넣는 것을 '밀어 넣는다' 하여 푸시(push)라고 하고 
반대로 넣어둔 자료를 꺼내는 것을 팝(pop)이라고 합니다.
예를 들어 1, 2, 3 순으로 스택에 자료를 넣고(push) 나서 
꺼내기 작업(pop)을 하면 3, 2, 1 순으로 나오게 됩니다.

1부터 N까지 정렬된 숫자가 들어있는 배열을 스택을 이용하여 순서를 바꿔 보려고 합니다. 
예를 들어,

1을 push하고 바로 pop → 1
2를 push, 3을 push하고, 2번 pop수행 → 3, 2
위와 같이 진행하면 1, 3, 2의 순서로 배열이 변경됩니다.

이렇게 스택을 이용하여 오름차순으로 정렬된 배열을 주어진 배열 arr로 바꿀 수 있는지 확인하는 solution 함수를 완성해 주세요.
'''

def solution(arr):
    pushed = [i for i in range(1, len(arr)+1)]
    stack=[]
    for i in pushed:
        stack.append(i)
        while stack and arr and stack[-1] == arr[0]:
            stack.pop()
            arr.pop(0)
    return not stack


    

s = [1, 3, 2]
print(solution(s))
# True

s1 = [3, 1, 2]	
print(solution(s1))
# False