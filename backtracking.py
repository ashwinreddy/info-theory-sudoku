import numpy as np

          
def find_empty_location(arr,l): 
    for row in range(9): 
        for col in range(9): 
            if(arr[row][col]==0): 
                l[0]=row 
                l[1]=col 
                return True
    return False
  

def used_in_row(arr,row,num): 
    for i in range(9): 
        if(arr[row][i] == num): 
            return True
    return False
  

def used_in_col(arr,col,num): 
    for i in range(9): 
        if(arr[i][col] == num): 
            return True
    return False
  
 
def used_in_box(arr,row,col,num): 
    for i in range(3): 
        for j in range(3): 
            if(arr[i+row][j+col] == num): 
                return True
    return False

def check_location_is_safe(arr,row,col,num): 
      
    
    return not used_in_row(arr,row,num) and not used_in_col(arr,col,num) and not used_in_box(arr,row - row%3,col - col%3,num) 
  

def solve_by_backtracking(arr, mode = "lower"): 
      
    l=[0,0] 
      
    if(not find_empty_location(arr,l)): 
        return True
      
    row=l[0] 
    col=l[1] 

    if mode == "lower":
        values = range(1,10)
    elif mode == "upper":
        values = range(9, 0, -1)
      
    for num in values:
          
        if(check_location_is_safe(arr,row,col,num)): 
              
            arr[row][col]=num 
  
            if(solve_by_backtracking(arr)): 
                return True
  
            arr[row][col] = 0
              
    return False 