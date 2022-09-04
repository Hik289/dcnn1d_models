
import sys
from exec import *

if __name__ == "__main__":

    all_cfg_paths = sys.argv[1:]
    for path in all_cfg_paths:
        
        parse(cfg_path=path)
        execute()



