print '--- Source ---'
file = open('data/glove/glove.840B.300d.txt')
for i in range(0,5):
  line  = file.readline()
  if line == "":
    break
  else:
    print(line)


print '--- Vocab ---'
file = open('data/glove/glove.840B.vocab')
for i in range(0,5):
  line  = file.readline()
  if line == "":
    break
  else:
    print(line)


