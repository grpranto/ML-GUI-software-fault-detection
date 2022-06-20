import math


def func(left, target, right):

    mid = math.floor((left + right) / 2)

    if right >= left:
        if target > arr[mid]:
            func(mid+1, target, right)
        elif target < arr[mid]:
            func(left, target, mid-1)
        else :
            print(str(target) + " is found")
    else :
        print(str(target) + " is not found")

print("what to search?")
search = int (input())

arr = [100, 55, 2, 3, 89, 50]
arr.sort()

func(0, search, 5)