from fsqr import *

def go():
    f = fsqr(main_path, edgelist_path)
    f.users_by_time(1,1)
    f.call_parallel()

if __name__ == '__main__':
    f = fsqr(main_path, edgelist_path)
    f.users_by_time(1,1)
    f.call_parallel()
    f.output()


f = fsqr(main_path, edgelist_path)
f.users_by_time(1,1)
a, b = f.user_analysis()
f.output()