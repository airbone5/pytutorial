import pandas

mydataset = {
  'cars': ["BMW", "Volvo", "Ford"],
  'passings': [3, 7, 2]
}

print(type(mydataset))
myvar = pandas.DataFrame(mydataset)

print(myvar)