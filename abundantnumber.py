n=int(input())
sum=1
for i in range(2,n):
    if n%i==0:
        sum=sum+i
if sum>n:
    print("it is a abundant number")
else:
    print("it is not a abundant number")
    