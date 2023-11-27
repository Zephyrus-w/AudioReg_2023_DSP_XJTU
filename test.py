import numpy as np
list=[
    [[1,2,3,4],[5,6,7,8]],
   [[9,8,7,6],[2,3,4,1]]
   ]
maximum=0
for i in range(2):
    for j in range(2):
        for k in range(len(list[i][j])):
            maximum= max(list[i][j][k], maximum)
    for j in range(2):
        for k in range(len(list[i][j])):
            list[i][j][k]/=maximum
        
print(list)
 