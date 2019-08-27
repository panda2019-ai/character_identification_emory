# coding:utf-8
"""

"""
import re

outfile = open('word_friends2.txt', 'wb')
with open('word_friends.txt', encoding='utf-8', errors='ignore') as infile:
    for line in infile:
        line = line.strip()
        matches = re.findall(u'([a-zA-Z\' \-\.]+)([^a-zA-Z\' \-\.。’`、0-9(:\sĴØ：\[_“ŤŤ：，üü\^ÿÿķ”ñ]+)', line)
        if matches and len(matches[0]) == 2:
            out_str = u'%s\t%s\n' % (matches[0])
            outfile.write(out_str.encode('utf-8', 'ignore'))
outfile.close()
