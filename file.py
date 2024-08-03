f = open('Task_2a.txt', 'r')
N = int(f.readline().split(" ")[2])
x = f.readline().split(" ")
x.pop()
for i in range(len(x)):
    x[i] = float(x[i])
f.close()