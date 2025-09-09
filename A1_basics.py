#CS 451/551
#Coding Practice Session 1
#Hanson, Kaitlyn 
#krhanson3@crimson.ua.edu


#Q1
import pandas as pd
dataset = {
    'Movies': ['50 First Dates', 'How to train your dragon', 
                'how to lose a guy in ten days', 'deadpool', 'iron man'],
    'Likes' : ['romantic', 'fantastic story', 'good plot', 'action packed', 'great actors'], 
    'Dislikes' : ['slow!', 'n/a', 'took too long......', 'too much language', 'sad']
}
df = pd.DataFrame(dataset)
# print(df)

#Q2
import string
df["Movies"]= df['Movies'].str.replace(" ", "")
edited_likes=[]
for s in df['Likes']:
    edited_string = ""
    for char in s:
        if char not in string.punctuation:
            edited_string += char
    edited_likes.append(edited_string)
df['Likes'] = edited_likes
edited_dislikes=[]
for s in df['Dislikes']:
    edited_string = ""
    for char in s:
        if char not in string.punctuation:
            edited_string += char
    edited_dislikes.append(edited_string)
df['Dislikes'] = edited_dislikes
# print (df)

#Q3
Unique_Word_Count = []

for index, row in df.iterrows():
    likes_words = row['Likes'].split()
    dislikes_words = row['Dislikes'].split()
    unique_words = len(set(likes_words + dislikes_words))
    Unique_Word_Count.append(unique_words)

df['Unique_Word_Count'] = Unique_Word_Count

print(df)





