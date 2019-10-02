# Author: cym

# 0-1背包问题。
# 每个物品i有价值v(i)，体积w(i)
# F(n, C)：将n个物品放进容量为C的背包，使得其价值最大
# F(i, c) = max(F(i-1, c), v(i)+F(i-1, c-w(i)))


class Solution:
    def __init__(self, n, C):
        # 实例变量. 给对象赋值
        self.memo = [[-1 for _ in range(C+1)] for _ in range(n)]


    def bestvalue(self, w, v, index, c):
        if index < 0 or c <= 0:
            return 0
        if self.memo[index][c] != -1:
            return self.memo[index][c]

        res = self.bestvalue(w, v, index-1, c)
        if c >= w[index]:
            res = max(res, v[index] + self.bestvalue(w, v, index-1, c-w[index]))
        self.memo[index][c] = res
        return res


    def knapsack01(self, w, v, C):
        return self.bestvalue(w, v, len(w)-1, C)


def knapsack01_d(w, v, C):
    n = len(w)
    if n == 0:
        return 0

    memo = [[-1 for _ in range(C + 1)] for _ in range(n)]
    for i in range(C+1):
        if i >= w[0]:
            memo[0][i] = v[0]
        else:
            memo[0][i] = 0

    for i in range(1, n):
        for j in range(C+1):
            memo[i][j] = memo[i-1][j]
            if j >= w[i]:
                memo[i][j] = max(memo[i][j], v[i] + memo[i-1][j-w[i]])

    return memo[n-1][C]


def knapsack01_d_lessmemo(w, v, C):
    n = len(w)
    if n == 0:
        return 0

    memo = [[-1 for _ in range(C + 1)] for _ in range(2)]
    for i in range(C+1):
        if i >= w[0]:
            memo[0][i] = v[0]
        else:
            memo[0][i] = 0

    for i in range(1, n):
        for j in range(C+1):
            memo[i % 2][j] = memo[(i-1) % 2][j]
            if j >= w[i]:
                memo[i % 2][j] = max(memo[i % 2][j], v[i] + memo[(i-1) % 2][j-w[i]])

    return memo[(n-1) % 2][C]

def knapsack01_d_best(w, v, C):
    n = len(w)
    if n == 0:
        return 0

    memo = [-1 for _ in range(C + 1)]
    for i in range(C+1):
        if i >= w[0]:
            memo[i] = v[0]
        else:
            memo[i] = 0

    for i in range(1, n):
        for j in range(C, w[i]-1, -1):
            memo[j] = max(memo[j], v[i] + memo[j-w[i]])

    return memo[C]


C = 5
w = [1, 2, 3]
v = [6, 10, 12]

s = Solution(len(w), C)
print(s.knapsack01(w, v, C))
print(knapsack01_d(w, v, C))
print(knapsack01_d_lessmemo(w, v, C))
print(knapsack01_d_best(w, v, C))
