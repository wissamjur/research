
l1 = [5746, 6835, 3567, 40491, 156605, 5490, 132333, 136850, 5764, 7899]
l2 = [6818, 5490, 132333, 8477, 40491, 156605, 25947, 67618, 57502, 3086]
# l2 = [5746, 5764, 6835, 7899, 67618, 3086, 6818, 136850, 3567, 5490]

# percentage similarity of lists
res = len(set(l1) & set(l2)) / float(len(set(l1) | set(l2))) * 100

print(res)