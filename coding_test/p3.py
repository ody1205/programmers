'''
주어진 문자열의 특정 구간을 뒤집는 연산을 하려고 합니다. 
다음은 문자열 S = abcde의 구간 [1, 3], [1, 4], [4, 5]를 
순서대로 뒤집는 예시입니다.
위 그림과 같이 문자열 abcde의 구간 [1, 3]을 뒤집으면 문자열은 cbade가 됩니다. 
다음으로 cbade의 구간 [1, 4]를 뒤집으면 문자열은 dabce가 됩니다. 
마지막으로 문자열 dabce의 구간 [4, 5]를 뒤집으면 dabec가 됩니다.
문자열 S와 뒤집어야 할 구간이 담긴 배열 interval이 매개변수로 주어질 때, 
주어진 문자열의 뒤집어야 할 구간을 모두 뒤집은 문자열을 return 하도록 
solution 함수를 완성해주세요.
'''

def solution(S, interval):
    for i in interval:
        to_flip = S[i[0]-1:i[1]]
        S = S[:i[0]-1] + to_flip[::-1] + S[i[1]:]



    return S

s = 'abcde'
i = [[1,3],[1,4],[4,5]]
print(solution(s,i))
# should print 'dabec'

# s1 = 'abcde'
# i1 = [[4,5],[1,2],[3,3]]
# print(solution(s1,i1))

