import os
list_dir = os.listdir('./data1')
import re
from soynlp.normalizer import repeat_normalize
import csv

pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

def clean(x):
    x = pattern.sub(' ', x)
    x = url_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x
datasets = []

with open('daum_blog_essay_sum_.csv', 'r', encoding='utf-8') as f:
    rdr = csv.reader(f)
    for line in rdr:
        line = line[0]
        line = line.replace('\xa0', ' ')
        line = line.replace('\\', '')
        line = line.replace('      ', '')
        line = line.replace('    ', ' ')
        line = line.replace('   ', ' ')
        line = line.replace('  ', ' ')
        line = line.replace('  ', ' ')
        line = line.replace('  ', ' ')
        line.strip()

        text = clean(line)#text.strip().replace("  "," ")

        if len(line)<70:
            continue
        print(line)
        datasets.append(line)
    f.close()

f = open('./data2/removed_sum.txt', 'r', encoding='utf-8')
while True:
    line = f.readline()
    datasets.append(line)
    if not line:
        break
f.close()

f = open('./data2/essay.txt', 'r', encoding='utf-8')
while True:
    line = f.readline()
    datasets.append(line)
    if not line:
        break
f.close()


f = open('train4.csv', 'w', newline='')
wr = csv.writer(f)
for data in datasets:
    data = data.replace('""', '"')
    data = data.replace('""', '"')
    data = data.replace('""', '"')
    data = data.replace('""', '"')
    data = data.replace('""', '"')
    data = data.replace('""', '"')
    data = clean(data)
    if len(data) <10:
        continue
    wr.writerow([clean(data)])

f.close()

 
# f = open('example.csv','r')
# rdr = csv.reader(f)
#
# for line in rdr:
#     print(line)
#
# f.close()

print(len(datasets))