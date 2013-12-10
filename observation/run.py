from fsqr import *

if __name__ == '__main__':
    f = fsqr(main_path, edgelist_path)
    f.users_by_time(1,1)
    f.call_parallel()