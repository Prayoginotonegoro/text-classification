import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 20.0
 
# Data to plot
#labels = 'hockey', 'movies', 'nfl', 'politics', 'soccer', 'worldnews', 'nba'
#sizes = [2, 22, 13, 213, 2, 179, 3]

#labels = 'hockey', 'movies', 'nfl', 'politics', 'soccer', 'nba', 'news'
#sizes = [2, 14, 0, 38, 10, 1, 108] 

labels = 'hockey', 'movies', 'nfl', 'worldnews', 'soccer', 'nba', 'news'
sizes = [3, 3, 3, 44, 2, 2, 211]
# Plot
colors=('b', 'g', 'r', 'c', 'm', 'y', 'burlywood')#, 'w')
plt.pie(sizes, labels=labels, colors = colors, autopct='%1.1f%%', startangle=140)
 
plt.axis('equal')
plt.show()
