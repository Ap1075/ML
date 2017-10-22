import re
exampleString = '''
Jack is 10 years old.
'''

names = re.findall(r'[A-Z][a-z]*', exampleString)
ages = re.findall(r'\d{1,3}',exampleString)

print(names)
print(ages)
