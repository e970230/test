#計算串列裡，所有偶數的乘積
lst = [1,9,0,3,6,3,2]
total = 1
for n in lst:
    if n%2 == 0 and n !=0:
        total*=n
print(total)