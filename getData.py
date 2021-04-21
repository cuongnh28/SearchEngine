from os import listdir
from os.path import isfile, join
from tika import parser
fileNames = [f for f in listdir("data") if isfile(join("data", f))]
documents = []
# print(allFiles)
for file in fileNames:
    raw = parser.from_file('data/' + file)
    documents.append(raw['content'])
print(fileNames)
print(documents)