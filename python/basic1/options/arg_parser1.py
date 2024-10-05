import argparse
parser = argparse.ArgumentParser()
parser.add_argument("arg1") # 字串
parser.add_argument("arg2",  nargs='?',default="b",help="第 2 個參數")# 字串
parser.add_argument("arg3",  nargs='?',default=0, help="第 3 個參數", type=int)
parser.add_argument("-v", "--verbose", help="比較多的說明", action="store_true")

args = parser.parse_args()
if args.verbose:
    print("這是測試結果")

print(f"第 1 個參數：{args.arg1:^10},type={type(args.arg1)}")
print(f"第 2 個參數：{args.arg2:^10},type={type(args.arg2)}")
print(f"第 3 個參數：{args.arg3:^10},type={type(args.arg3)}")