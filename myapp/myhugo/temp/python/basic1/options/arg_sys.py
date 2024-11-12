# demo python hello.py --option 1
import sys
 

def main(): #Note1: def main1()
    # standard - load "args" list with cmd-line-args
    args = sys.argv[0:] #冒號後面是空的代表從0到最後
    print(args)



if __name__ == '__main__':  
    main() # 如果Note1的函數是main1,那這裡就是main1(),不是一定要main()