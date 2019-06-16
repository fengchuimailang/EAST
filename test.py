# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        if len(array) == 0:
            return False
        i = 0
        j = 0
        width = len(array[0])
        height = len(array)
        p = array[0][0]
        while j + 1 < width and p < target:
            j += 1
            p = array[i][j]
        if p == target:
            return True
        elif j == 0:
            return False
        else:
            j -= 1
            p = array[i][j]
        while i + 1 < height and p < target:
            i += 1
            p = array[i][j]
        if p == target:
            return True
        return False


if __name__ == "__main__":
    target = 15
    array = [[1, 2, 8, 9], [2, 4, 9, 12], [4, 7, 10, 13], [6, 8, 11, 15]]
    s = Solution()
    print(s.Find(target, array))
