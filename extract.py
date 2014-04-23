from __future__ import print_function
import os
import sys
import re
import commands

"""
Usage python extract.py <in_dir> <out_dir>
"""

def main():
   name_pattern = re.compile('(.*)\.pdf')
   in_dir = sys.argv[1]
   out_dir = sys.argv[2]
   for pdf in os.listdir(in_dir):
       name_match = name_pattern.match(pdf)
       if name_match:
           save_name = os.path.join(out_dir, name_match.group(1)+'.txt')
           cmd = 'pdf2txt.py -o '+save_name+' '+os.path.join(in_dir, pdf)
           (status, output) = commands.getstatusoutput(cmd)




if __name__ == '__main__':
    main()

