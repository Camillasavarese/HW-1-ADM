#What's your name
<<<<<<< HEAD
print("Hello, World!")
#Python IF-Else
import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
if n % 2 == 0:
    if n in range(2,6):
        print("Not Weird")

    elif n in range(6,21):
        print("Weird")

    elif n > 20:
        print("Not Weird")
else:
    print("Weird")
#Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)
#Python:Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)
#Loops
if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i**2)
#Write a function
def is_leap(year):
    leap = False
    return  year % 4 == 0 and (year % 400 == 0 or year % 100 != 0)
#Print Function
if __name__ == '__main__':
    n = int(input())
    print(*range(1, n+1), sep='')
#List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    lista = [[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i + j + k != n ]
    print(lista)
#Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr= list(map(int, input().split()))
    massimo=max(arr)
    while max(arr)== massimo:
        arr.remove(massimo)
    print(max(arr))
#Nested Lists
if __name__ == '__main__':
    d={}
    for _ in range(int(input())):
        name = input()
        score = float(input())
        d[name]=score 
    score_due=d.values()
    sec=sorted(list(set(score_due)))[1] 
    second_lowest=[] 
    for key,value in d.items(): 
        if value==sec:
            second_lowest.append(key)
    second_lowest.sort() 
    for i in second_lowest:
                     print(i )
            
#Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
query_scores = student_marks[query_name] #prendo i valori che mi servono
total_scores = sum(query_scores)
avg = total_scores/3
print('%.2f' % avg)
#Lists
if __name__ == '__main__':
    N = int(input())
    l = [];
    for i in range(0,N):
        a = input().split()
        if a[0] == "print":
            print(l)
        elif a[0] == "insert":
            l.insert(int(a[1]),int(a[2]))
        elif a[0] == "remove":
            l.remove(int(a[1]))
        elif a[0] == "pop":
            l.pop();
        elif a[0] == "append":
            l.append(int(a[1]))
        elif a[0] == "sort":
            l.sort();
        else:
            l.reverse();
#Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t=tuple(integer_list)
    print(t)
#sWAP cASE
def swap_case(s):
    return s.swapcase()
#String Split and Join
def split_and_join(line):
    c= line.split(" ")
    d= "-".join(c)
    return d
#What's your name?
def print_full_name(first, last):
    print('Hello {} {}! You just delved into python.'.format(first, last))
#Mutations
def mutate_string(string, position, character):
    l = list(string)
    l[position] = character
    string = ''.join(l)
    return string
#Find a string
def count_substring(string, sub_string):
    count = 0
    for i in range(len(string)):
        if string[i:].startswith(sub_string):
            count += 1
    return count
#String Validators
if __name__ == '__main__':
    s = input()
    print(any(char.isalnum() for char in s))
    print(any(char.isalpha() for char in s))
    print(any(char.isdigit() for char in s))
    print(any(char.islower() for char in s))
    print(any(char.isupper() for char in s))

#Text Alignment
thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))
#Text Wrap
def wrap(string, max_width):
    return  "\n".join(textwrap.wrap(string, max_width))
#Designer Door Mat
N, M = map(int,input().split())
for i in range(1,N,2): 
    print((i * ".|.").center(M, "-"))
print("WELCOME".center(M,"-"))
for i in range(N-2,-1,-2): 
    print((i * ".|.").center(M, "-"))
#String Formatting
def print_formatted(number):
    width=len(bin(number)[2:])
    for i in range(1,number+1):
        deci=str(i)
        octa=oct(i)[2:]
        hexa=(hex(i)[2:]).upper()
        bina=bin(i)[2:]
        print(deci.rjust(width),octa.rjust(width),hexa.rjust(width),bina.rjust(width))
#Alphabet Rangoli
def print_rangoli(size):
    alp = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(size-1,-size,-1):
        temp = '-'.join(alp[size-1:abs(i):-1]+alp[abs(i):size])
        print(temp.center(4*size-3,'-'))
#Capitalize!
def solve(s):
    return' '.join(word.capitalize() for word in s.split(' '))
#The Minion Game
def minion_game(string):
    vocali='AEIOU'
    kevin=0
    stuart=0
    for i in range(len(s)):
        if s[i] in vocali:
            kevin += (len(s)-i)
        else:
            stuart += (len(s)-i)
    if kevin > stuart:
        print ("Kevin", kevin)
    elif kevin < stuart:
         print ("Stuart", stuart)
    else:
        print ("Draw")
#Merge the Tools!
def merge_the_tools(string, k):
    for i in range(0, len(string), k):
        s = ""
        for j in string[i : i + k]:
            if j not in s:
                s += j          
        print(s)
#Introduction to Sets
def average(array):
    avg=sum(set(array))/len(set(array))
    return avg
#No Idea!
n, m = input().split()
ar = input().split()
a = set(input().split())
b = set(input().split())
cont=0 
for i in ar:
    if i in a:
        cont+=1
    elif i in b:
        cont-=1
    else:
        cont+=0
print(cont)
#Symmetric Difference
a,b=(int(input()),input().split())
c,d=(int(input()),input().split())
x=set(b)
y=set(d)
p=y.difference(x)
q=x.difference(y)
print ('\n'.join(sorted(p.union(q), key=int)))
#Set.add()
n= int(input())
stampe= set()
for i in range(n):
    stampe.add(input())
print(len(stampe))
#Set.discard(),.remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
N=int(input())
for i in range(N) :
    choice=input().split()
    if choice[0]=="pop" :
        s.pop()
    elif choice[0]=="remove" :
        s.remove(int(choice[1]))
    elif choice[0]=="discard" :
        s.discard(int(choice[1]))
print (sum(s))   
#Set.union()
n=int(input())
a= set(input().split())
m=int(input())
b= set(input().split())
print(len(a.union(b)))
#Set.intersection()
n=int(input())
a= set(input().split())
m=int(input())
b= set(input().split())
print(len(a.intersection(b)))
#Set.difference()
n=int(input())
a= set(input().split())
m=int(input())
b= set(input().split())
print(len(a.difference(b)))
#Set.symmetric_difference()
n=int(input())
a= set(input().split())
m=int(input())
b= set(input().split())
c=a.difference(b)
d=b.difference(a)
print(len(c.union(d)))
#Set Mutations
len_set = int(input())
storage = set(map(int, input().split()))
op_len = int(input())
for i in range(op_len):
    operation = input().split()
    if operation[0] == 'intersection_update':
        temp_storage = set(map(int, input().split()))
        storage.intersection_update(temp_storage)
    elif operation[0] == 'update':
        temp_storage = set(map(int, input().split()))
        storage.update(temp_storage)
    elif operation[0] == 'symmetric_difference_update':
        temp_storage = set(map(int, input().split()))
        storage.symmetric_difference_update(temp_storage)
    elif operation[0] == 'difference_update':
        temp_storage = set(map(int, input().split()))
        storage.difference_update(temp_storage)
print(sum(storage))

#The Captain's Room
k=int(input())
lista= list(input().split())
s1=set();  #una sola volta
s2=set();  #piu di una
for i in lista:
    if  i in s1:
        s2.add(i);
    else:
        s1.add(i);
s3=s1.difference(s2);
print(list(s3)[0])
#Check Strict Superset
A = set(input().split())

for _ in range(int(input())):
    if not A.issuperset(set(input().split())):
        print(False)
        break
        
else:
    print(True)
#Check Superset
for i in range (int(input())):
    _, a = input(), set(input().split())
    _, b = input(), set(input().split())
    print(b.intersection(a) == a)
#collections.Counter()
from collections import Counter
X= int(input())
sizes= Counter(map(int, input().split()))
N= int(input())
soldi=0
for i in range(N):
    size, price = map(int, input().split())
    if sizes[size]>0: 
        soldi += price
        sizes[size] -= 1

print (soldi)

#DefaultDict Tutorial
from collections import defaultdict
A = defaultdict(list) #tipo lista vuota

n, m = map(int,input().split())
for i in range(1, n+1):
    A[input()].append(str(i))
    
for i in range(m):
    b=input()
    if b in A:
        print(' '.join(A[b]))
    else:
        print(-1)
#Collections.namedtuple()
from collections import namedtuple
n, Student = int(input()), namedtuple('Student', input())
print("{:.2f}".format(sum([int(Student(*input().split()).MARKS) for _ in range(n)]) / n))

#Collections.OrderedDict()
from collections import OrderedDict
N = int(input())
d = OrderedDict()
for i in range(N):
    item = input().split()
    itemPrice = int(item[-1])
    itemName = " ".join(item[:-1])
    if(d.get(itemName)):
        d[itemName] += itemPrice
    else:
        d[itemName] = itemPrice
for i in d.keys():
    print(i, d[i])
#Word Order
from collections import OrderedDict
vuoto = OrderedDict()
n=int(input())
for _ in range(n):
    word = input()
    vuoto.setdefault(word, 0)
    vuoto[word] += 1
   
print(len(vuoto))
print(*vuoto.values())

#Collections.deque()
from collections import deque
d = deque()
N= int(input())
for i in range(N):
    a=input().split()
    if a[0]=="append":
        d.append(int(a[1]))
    elif a[0]=="pop":
        d.pop()
            
    elif a[0]=="popleft":
        d.popleft()
    elif a[0]=="appendleft":
        d.appendleft(a[1])
print(*d)
#Company Logo
import math
import os
import random
import re
import sys
from collections import OrderedDict
from collections import Counter

if __name__ == '__main__':
    s = input()
    s_1=list(s)
    c = Counter(s_1).most_common()  
    c = sorted(c, key=lambda x: (-x[1] , x[0]))
    #print(c)
    for i in range(0, 3):
        print(c[i][0], c[i][1])


#Piling Up!
from collections import deque
def pill(d):
    while d:
        b = d.popleft() if d[0]>d[-1] else d.pop()
        if not d:
            return "Yes"
        if d[-1]>b or d[0]>b:
            return "No"
T = int(input())
for i in range(T):
    
    N = int(input())
    b_1 = deque(map(int, input().split()), N)
    print(pill(b_1))
#Calendar Module
import calendar
m,d,y = map(int,input().split())

print(calendar.day_name[calendar.weekday(y, m, d)].upper())
#Time delta
import math
import os
import random
import re
import sys
import datetime

import math
import os
import random
import re
import sys
from datetime import datetime
# Complete the time_delta function below.
def time_delta(t1, t2):
    time_format = '%a %d %b %Y %H:%M:%S %z'
    t1 = datetime.strptime(t1, time_format)
    t2 = datetime.strptime(t2, time_format)
    return str(int(abs((t1-t2).total_seconds())))   

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()
        #print(t1)

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()
#Exceptions
T= int(input())
for i in range(T):
    try:
        a,b= map(int,input().split())
        print (a//b)
    except ZeroDivisionError :
        print("Error Code:","integer division or modulo by zero");
    except ValueError as v:
        print("Error Code:",v);
#Zipped!
n, x = map(int, input().split()) 
sheet = []
for _ in range(x):
    sheet.append( map(float, input().split()) ) 

for i in zip(*sheet): 
    print( sum(i)/len(i))
#Athlete Sort
import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    
    s = sorted(arr, key = lambda x: x[k])
    for i in s:
        print(*i, sep=' ')
#ginortS
s=input()
print(*(sorted(s, key=lambda x: (x.isdigit(), x.isdigit() and int(x)%2==0, x.isupper(), x.islower(), x))), sep='')
#Map and Lambda Function
cube = lambda x: x**3# complete the lambda function 

def fibonacci(n):
    l= [0,1]
    
    for i in range(2,n):
        l.append(l[i-1]+l[i-2])
    return(l[0:n])
#Detect Floating Number
import re
T=int(input())
for i in range(T):
    
 print (bool(re.search(r'^[+-]?[0-9]*\.[0-9]+$',input())))

#Re.split()
regex_pattern = r"[.,]"
#Group(),Groups() & Groupdict()
import re
s=input()
expression=r"([a-zA-Z0-9])\1+"

m = re.search(expression,s)

if m:
    print(m.group(1))
else:
    print(-1)
#Re.findall() & Re.finditer()
s=input()
import re
consonants = 'qwrtypsdfghjklzxcvbnm'
vowels = 'aeiou'
expr = re.findall(r'(?<=['+consonants+'])(['+vowels+']{2,})(?=['+consonants+'])',s,flags = re.I)
if expr:
    for i in expr:
        print (i)
else:
    print (-1)
#Re.start() & Re.end()
import re
s=input()
k=input()



k = re.compile(k)
match = k.search(s)

if not match: 
    print('(-1, -1)')
    
while match:
    print('({0}, {1})'.format(match.start(), match.end() - 1))
    match = k.search(s, match.start() + 1)

#Regex Substitution
import re


S=int(input())
for i in range(S):
      x=re.compile(r'(?<= )(&&)(?= )')
      y=re.compile(r'(?<= )(\|\|)(?= )')
      
      print(y.sub('or', x.sub('and', input())))

#Validating Roman Numerals
import re


thousand = 'M{0,3}'
hundred = '(C[MD]|D?C{0,3})'
ten = '(X[CL]|L?X{0,3})'
digit = '(I[VX]|V?I{0,3})'
regex_pattern= thousand + hundred+ten+digit +'$'
#Validating Phone numbers
import re
N=int(input())
for i in range(N):
    if re.match(r'[789]\d{9}$',input()):   
        print("YES" ) 
    else:  
        print ("NO"  )
#Validating and Parsing Email Address
import re
n = int(input())
for _ in range(n):
    name, mail = input().split()
    m = re.match(r'<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>', mail)
    if m:
        print(name,mail)

#Hex Color 
import re
N=int(input())
for _ in range(N):
    
    s=input()
    x=s.split()
    
    if '{' not in x and len(x)>1 :
        match=re.findall('#[0-9a-fA-F]{3,6}', s)
        [print(i) for i in match]
#HTML Parser 1
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print ('Start :', tag)
        for ele in attrs:
            print ('->', ele[0], '>', ele[1])

    def handle_endtag(self, tag):
        print ('End   :', tag)

    def handle_startendtag(self, tag, attrs):
        print ('Empty :', tag)
        for ele in attrs:
            print ('->', ele[0], '>', ele[1])

p = MyHTMLParser()
for _ in range(int(input())):
    p.feed(input())
#HTML Parser 2
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if (len(data.split('\n')) != 1):
            print(">>> Multi-line Comment")
        else:
            print(">>> Single-line Comment")
        print(data.replace("\r", "\n"))
    def handle_data(self, data):
        if data!= "\n":
            
            print(">>> Data")
            print(data)
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()
#Detect HTML tags and Attributes
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print("->", attr[0], ">", attr[1])

p = MyHTMLParser()
for _ in range(int(input())):
    p.feed(input())
#Validating UID
import re
N=int(input())
for i in range(N):
    if re.match(r'^(?!.*(.).*\1)(?=(?:.*[A-Z]){2,})(?=(?:.*\d){3,})[\w]{10}$',input()):   
        print("Valid" ) 
    else:  
        print ("Invalid"  )
#Validating Credit Card number
import re
p = re.compile(
    r"^" 
    r"(?!.*(\d)(-?\1){3})"
    r"[456]"
    r"\d{3}"
    r"(?:-?\d{4}){3}"
    r"$")
for _ in range(int(input().strip())):
    print("Valid" if p.search(input().strip()) else "Invalid")
#Validating Postal Code
regex_integer_in_range = r"^[0-9][\d]{5}$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"	
#Matrix Script
import math
import os
import random
import re
import sys

first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)
 #ruoto   
matrix= list(zip(*matrix))
#print(matrix)

sample=str()
for words in matrix:
    for char in words:
        sample += char
#print(sample)
       
print(re.sub(r'(?<=\w)([^\w\d]+)(?=\w)', ' ', sample)) 
#XML 1
def get_attr_number(node):
    return(len(node.attrib)+ sum(get_attr_number(i)for i in node))
#XML 2
maxdepth = 0
def depth(elem, level):
    global maxdepth
    level += 1
    
    if len(elem) > 0:
        for tag in elem:
            depth(tag, level)
    else:
        if maxdepth < level:
            maxdepth = level
#Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        f(['+91 ' + c[-10:-5] + ' ' + c[-5:] for c in l])
    
        
        # complete the function
    return fun
#Decorators 2 - Name Directory
from operator import itemgetter
def person_lister(f):
    def inner(people):
        # complete the function  
        return map(f, sorted(people, key=lambda x: int(x[2])))       
    return inner
#Arrays
def arrays(arr):
    return numpy.array(arr[::-1],float)
#Shape and Reshape
import numpy

my_array = numpy.array(input().split(),int)
print (numpy.reshape(my_array,(3,3)))

#Transpose and Flatten
import numpy
n, m = map(int, input().split())
array = numpy.array([input().strip().split() for _ in range(n)], int)
print (array.transpose())
print (array.flatten())
#Concatenate
import numpy

n, m,p = map(int, input().split())
array_1 = numpy.array([input().split() for _ in range(n)], int)
array_2 = numpy.array([input().split() for _ in range(m)], int)
print (numpy.concatenate((array_1, array_2), axis = 0))
#Zeros and Ones
import numpy
shape = list(map(int, input().split()))
print (numpy.zeros(shape, dtype = numpy.int))
print (numpy.ones(shape, dtype = numpy.int))
#Eye and Identity
import numpy
numpy.set_printoptions(legacy='1.13')
n, m = map(int, input().split())
print (numpy.eye(n, m))
#Array mathematics
import numpy

n, m = map(int, input().split())
a = numpy.array([input().strip().split() for _ in range(n)], int)
b=numpy.array([input().strip().split() for _ in range(n)], int)

print (numpy.add(a, b)  )         
print (numpy.subtract(a, b)     )

print (numpy.multiply(a, b)   )    
print (numpy.array(a/b,int))   
print (numpy.mod(a, b)     )     


print (numpy.power(a, b)    )     
#Floor,Ceil and Rint
import numpy
numpy.set_printoptions(legacy='1.13')
a = numpy.array(input().split(),float)

print(numpy.floor(a))
print(numpy.ceil(a))
print(numpy.rint(a))
#Sum and Prod
import numpy

n, m = map(int, input().split())
array = numpy.array([input().split() for _ in range(n)], int)

somma= numpy.sum(array, axis = 0)   
print(numpy.prod(somma)     )     
#Min and Max
import numpy

n, m = map(int, input().split())
array = numpy.array([input().split() for _ in range(n)], int)
mini=numpy.min(array, axis = 1)

print (numpy.max(mini))
#Mean,Var and Std
import numpy
numpy.set_printoptions(legacy='1.12')
n, m = map(int, input().split())
array = numpy.array([input().split() for _ in range(n)], int)
print (numpy.mean(array, axis = 1))
print (numpy.var(array, axis = 0) )   
print(round(numpy.std(array), 11))
#Dot and Cross
import numpy
n=int(input())
a = numpy.array([input().split() for _ in range(n)],int)
b = numpy.array([input().split() for _ in range(n)],int)
print(numpy.dot(a,b))
#Inner and Outer
import numpy
A = numpy.array(input().split(), int)
B = numpy.array(input().split(), int)
print(numpy.inner(A, B), numpy.outer(A, B), sep='\n')
#Polynomials
import numpy
p = list(map(float, input().split()))
print(numpy.polyval(p, float(input())))
#Linear Algebra
import numpy
n=int(input())
A = numpy.array([input().split() for _ in range(n)], float)
det= numpy.linalg.det(A)
print(round(det, 2))
#BirthdayCakeCandles
import math
import os
import random
import re
import sys

def birthdayCakeCandles(candles):
    
    max_value = max(candles)
    return (candles.count(max_value))
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()
#Number Line Jumps
import math
import os
import random
import re
import sys
def kangaroo(x1, v1, x2, v2):
    if x1 < x2 and v1 < v2:
        return 'NO'    
    
    else:
        if v1!=v2 and (x2-x1)%(v2-v1)==0:
            return 'YES' 
        else:
            return 'NO'

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()
#Viral Advertising
import math
import os
import random
import re
import sys
from math import floor

def viralAdvertising(n):
    shared=5
    liked=0
    for i in range(1,n+1):
        liked+= math.floor(shared/ 2)
        shared= math.floor(shared/ 2)*3
        
    return liked
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()
#Recursive Digit Sum
import math
import os
import random
import re
import sys

def superDigit(n, k):
    x = n * k % 9
    n=int(n)
    return x if x else 9
    # Write your code here

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]
    n=int(n)

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()
#Insertion Sort 1
import math
import os
import random
import re
import sys

def insertionSort1(n, arr):
    last=arr[-1]
    i=n-1
    while i>0 and arr[i-1]>last:
        arr[i]=arr[i-1]
        print(*arr)
        i-=1
    arr[i]=last
    print(*arr)
 
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)
#Insertion Sort 2
import math
import os
import random
import re
import sys

def insertionSort2(n, a):
    for j in range(1,n):
        key=a[j]
        i=j
        while i>0 and a[i-1]>key:
            a[i]=a[i-1]
            i-=1
        a[i]=key
        print(*a)
  
if __name__ == '__main__':
    n = int(input().strip())

    a = list(map(int, input().rstrip().split()))

    insertionSort2(n, a)

#Note: I look at HackerRanK discussion in the section "Regex and Parsing","XML" and for the challenge "Recursive Digit Sum"


=======
print("Hello, World!")
>>>>>>> main
