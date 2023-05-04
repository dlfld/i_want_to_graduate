from itertools import product,permutations

if __name__ == '__main__':
    list1 = [1, 2, 3, 4]
    list2 = [1, 2, 3, 4]
    res = list(product(list1, repeat=2))
    for item in res:
        print(list(item)+[1])
