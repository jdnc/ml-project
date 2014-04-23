from __future__ import print_function
import os
import sys
from collections import defaultdict
import json
import re
import pandas


"""
Usage python get_data.py <in_dir>
"""
pattern = re.compile('(.*)\.txt')

def get_data(in_dir):
    with open('data/terms.json', 'rb') as f:
        terms = json.load(f)
    row_dict = {}
    for text in os.listdir(in_dir):
        file_match = pattern.match(text)
        if file_match:
            word_count = 0.0
            column_dict = defaultdict(int)
            f = open(os.path.join(in_dir, text), 'rb')
            lines = f.readlines()
            for line in lines:
                words = [word.lower() for word in line.split()]
                word_count += len(words)
                for word in words:
                    if word.decode('utf-8') in  terms:
                        column_dict[word] += 1
            for key in column_dict:
                column_dict[key] = float(column_dict[key])/float(word_count)
            row_key = file_match.group(1).replace('_', '/')
            row_dict[row_key] = column_dict
    return row_dict


def store_data(row_dict, out_file):
    df = pandas.DataFrame.from_dict(row_dict, orient='index')
    df.to_csv(out_file)

def main():
    in_dir = sys.argv[1]
    row_dict = get_data(in_dir)
    store_data(row_dict, 'data/new_database.txt')


if __name__ == '__main__':
    main()
