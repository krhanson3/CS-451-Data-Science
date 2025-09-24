#Python Review 1 (If-Then & Lists)
# Kaitlyn Hanson 

#A1
import random
print(random.randint(0,100))

#A2
import random
random_int = random.randint(0,100)
if random_int > 50:
    print('bigger than 50')
else:
    print('smaller than 50')

#A3
import random
num = random.randint(0, 1000)

def character(num):
    if num < 250:
        return str(num) + ': smaller than 250'
    elif 250 <= num < 500:
        return str(num) + ': bigger than or equal to 250, but smaller than 500'
    else:  # num >= 500
        return str(num) + ': bigger than or equal to 500'

print(character(num))

#A4
import random 
x = 1
while x <= 5:
    num = random.randint(0, 1000)
    print(character(num))
    x += 1

#B1
my_list = [2.72, 3.14, 55, 9, 110, 'To be or not to be', 11, 132, 88, 1.1]

print(my_list[1])
print(my_list[3])

print(my_list[1:6])

#B2
my_list.remove('To be or not to be')
print(my_list)

#B3
my_list.sort()
print(my_list)