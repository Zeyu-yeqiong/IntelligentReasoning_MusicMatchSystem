
nums=['1 3','3 5','2 4']
        
res=0
for i in range(len(nums)):
    for j in range(i,len(nums)):
        area=abs(int(nums[i][0])-int(nums[j][0]))
        print(area)
        if area>res:
            res=area
            
            
print(res)