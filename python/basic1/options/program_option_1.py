# 執行
# python program_option_1.py
# python program_option_1.py -option 1

import sys
 

def main():
    # standard - load "args" list with cmd-line-args
    args = sys.argv[0:]
    print(args)
   

# Python boilerplate.
if __name__ == '__main__':
    print('start')
    main()
