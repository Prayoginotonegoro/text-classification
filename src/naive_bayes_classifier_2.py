import math
import numpy as np
import matplotlib.pyplot as plt


def words_from_file(file):
    """this function takes a file as input and returns a list of words """
    file_handle = open(file, "r")
    content = file_handle.read()  # create a string from the file
    file_handle.close()
    lst = content.split()
    return lst


def list_from_file(file):
    """this function takes a file as input and returns a list of strings """
    file_handle = open(file, "r")
    content = file_handle.readlines()  # create a string from the file
    file_handle.close()
    data_set_list = []
    for line in content:
        temp_list = line.splitlines()
        data_set_list.append(temp_list)
    return data_set_list


def words_from_list(lst):
    """this function takes a list of strings as input and returns a list of all words inside that given list"""
    result = []
    for v in lst:
        temp_list = v[0].split()
        result.extend(temp_list)
    return result


def category_list_maker():
    """this function takes training input file and returns records of each of 8 classes as a list"""
    hockey_list = []
    movies_list = []
    nba_list = []
    news_list = []
    nfl_list = []
    politics_list = []
    soccer_list = []
    worldnews_list = []
    for index, category in enumerate(train_output_list):
        if category == "hockey":
            hockey_list.append(train_input_list[index])
        elif category == "movies":
            movies_list.append(train_input_list[index])
        elif category == "nba":
            nba_list.append(train_input_list[index])
        elif category == "news":
            news_list.append(train_input_list[index])
        elif category == "nfl":
            nfl_list.append(train_input_list[index])
        elif category == "politics":
            politics_list.append(train_input_list[index])
        elif category == "soccer":
            soccer_list.append(train_input_list[index])
        elif category == "worldnews":
            worldnews_list.append(train_input_list[index])
    return hockey_list, movies_list, nba_list, news_list, nfl_list, politics_list, soccer_list, worldnews_list

def frequency_counter(xs, ys):
    """this function takes a list of words and a dictionary as input and returns a list that corresponds to
    the frequency of each word in dictionary that appears in the list """
    xs.sort()  # this is sorted input list
    #  creating a list with the length of dictionary(ys) and fill it with 0
    counter_list = []
    for i in range(len(ys)):
        counter_list.append(0)
    xi = 0
    yi = 0

    while True:
        if xi >= len(xs):
            break
        if yi >= len(ys):
            break
        if xs[xi] == ys[yi]:
            # increment frequency of the corresponding word in dictionary
            counter_list[yi] += 1
            xi += 1
        elif xs[xi] < ys[yi]:
            xi += 1
        else:
            yi += 1
    return counter_list


def probability_calculator(frequency_list, word_list):
    """this function calculate probability of each word in dictionary given class.
    in other word P(word_k|class) = (n_k + 1)/ (n + len(dictionary)) in which, k: {words in dictionary},
    n_k: number of time word k in dictionary appears in class' word set, n: number of words in class' word set"""
    result = []
    for i,word in enumerate(dictionary):
        p_word_given_class = (frequency_list[i] + 1)/(len(word_list) + len(dictionary))
        result.append(p_word_given_class)
    return result


def joint_probability(lst1, lst2):
    # print(lst1)
    # print(lst2)
    result = 0
    # i = 0
    for i, v in enumerate(lst2):
        if lst1[i] != 0:
            result += math.log(v) * int(lst1[i])
        # i += 1
    return result


def prediction(lst):
    category_list = ["hockey", "movies", "nba", "news", "nfl", "politics", "soccer", "worldnews"]
    probability_list = []
    class_conditional_probability_list = [p_words_given_hockey, p_words_given_movies, p_words_given_nba,
                                          p_words_given_news, p_words_given_nfl, p_words_given_politics,
                                          p_words_given_soccer, p_words_given_worldnews]
    class_probability_list = [p_hockey, p_movies, p_nba, p_news, p_nfl, p_politics, p_soccer, p_worldnews]
    temp_list = lst[0].split()
    feature_vector = frequency_counter(temp_list, dictionary)
    # print(feature_vector)
    for i, v in enumerate(class_conditional_probability_list):
        probability_list.append(joint_probability(feature_vector, v) +
                                math.log(class_probability_list[i]))
    index = probability_list.index(max(probability_list))
    return category_list[index]


def file_maker2(name, lst):
    import csv
    with open(name, 'w') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        file_writer.writerow(['id', 'category'])
        for i, row in enumerate(lst):
            file_writer.writerow([i, row])
    return None


def duplicate_remover(lst):
    """this remove duplicate from a list"""
    new_lst = []
    recent_element = None
    for element in lst:
        if element != recent_element:
            new_lst.append(element)
            recent_element = element
    return new_lst


def file_maker(name, lst):
    import csv
    with open(name, 'w') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        file_writer.writerow(['id', 'category'])
        for i, row in enumerate(lst):
            file_writer.writerow([i, row])
    return None


def dictionary_maker(xs, ys, size):
    result = []
    counter_list = []
    for i in range(len(ys)):
        counter_list.append(0)
    xi = 0
    yi = 0
    # result = []

    while True:
        if xi >= len(xs):
            # result.extend(ys[yi:])
            break
        if yi >= len(ys):
            # result.extend(xs[xi:])
            break
        if xs[xi] == ys[yi]:
            # result.append(xs[xi])
            counter_list[yi] += 1
            xi += 1
            # yi += 1
        elif xs[xi] < ys[yi]:
            # result.append(xs[xi])
            xi += 1
        else:
            # result.append(ys[yi])
            yi += 1
        # print(result, yi, len(ys), xi)
    # return counter_list

    for i, v in enumerate(counter_list):
        if v > size:
            result.append(ys[i])
    return result


all_data_input_list = list_from_file("train_input_processed_1_grams.csv")
all_data_output_list = list_from_file("train_output_processed.csv")
training_input_list = all_data_input_list[:int(len(all_data_input_list)*.8)]
training_output_list = all_data_output_list[:int(len(all_data_output_list)*.8)]
validation_input_list = all_data_input_list[int(len(all_data_input_list)*.8):]
validation_output_list = all_data_output_list[int(len(all_data_output_list)*.8):]

training_data_set = words_from_list(training_input_list)
train_output_list = words_from_list(training_output_list)
train_input_list = training_input_list
val_output_list = words_from_list(validation_output_list)
# for lst in  train_input_list[:100]:
#     print(lst)
dictionary = words_from_file("dictionary_10000_lamin.csv")
# test_input_list = list_from_file("test_input_processed.csv")


##################################################################################################
# making dictionary ### Uncomment this section to create dictionaries ###
# training_data_set.sort()
# main_dictionary = duplicate_remover(training_data_set)
# dictionary_10000_lamin = dictionary_maker(training_data_set, main_dictionary, 35)
# print(len(dictionary_10000_lamin))
# file_maker("dictionary_10000_lamin.csv", dictionary_10000_lamin)

##################################################################################################

# creating a list of records for each class
hockey_list, movies_list, nba_list, news_list, nfl_list, politics_list, soccer_list, worldnews_list \
    = category_list_maker()



# finding the probability of each category(class)
p_hockey = len(hockey_list)/len(train_output_list)
p_movies = len(movies_list)/len(train_output_list)
p_nba = len(nba_list)/len(train_output_list)
p_news = len(news_list)/len(train_output_list)
p_nfl = len(nfl_list)/len(train_output_list)
p_politics = len(politics_list)/len(train_output_list)
p_soccer = len(soccer_list)/len(train_output_list)
p_worldnews = len(worldnews_list)/len(train_output_list)

category_list = ["hockey", "movies", "nba", "news", "nfl", "politics", "soccer", "worldnews"]
category_probability = [p_hockey, p_movies, p_nba, p_news, p_nfl, p_politics, p_soccer, p_worldnews]


def bar_chart_plotter(xs, ys):
    """this function takes the list of different class size and plot them"""
    n = 8
    ind = np.arange(n)  # the x locations for the groups
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()
    rects = ax.bar(ind, ys, width, color='b')
    ax.set_xticklabels(xs, rotation=45)
    ax.set_ylabel('percent')
    ax.set_title('percentage of examples in training set for each category')
    ax.set_xticks(ind + width / 2)
    plt.axis([-.1, 7.5, 0, .2])
    
    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                    '%.2f' % (height*100),
                    ha='center', va='bottom', rotation='vertical')

    autolabel(rects)
    plt.show()

    return None

bar_chart_plotter(category_list, category_probability)


# finding all words for each category and putting them into a single list of words
hockey_words_list = words_from_list(hockey_list)
movies_words_list = words_from_list(movies_list)
nba_words_list = words_from_list(nba_list)
news_words_list = words_from_list(news_list)
nfl_words_list = words_from_list(nfl_list)
politics_words_list = words_from_list(politics_list)
soccer_words_list = words_from_list(soccer_list)
worldnews_words_list = words_from_list(worldnews_list)

# finding the frequency of each word in dictionary based on each category
hockey_words_frequency = frequency_counter(hockey_words_list, dictionary)
movies_words_frequency = frequency_counter(movies_words_list, dictionary)
nba_words_frequency = frequency_counter(nba_words_list, dictionary)
news_words_frequency = frequency_counter(news_words_list, dictionary)
nfl_words_frequency = frequency_counter(nfl_words_list, dictionary)
politics_words_frequency = frequency_counter(politics_words_list, dictionary)
soccer_words_frequency = frequency_counter(soccer_words_list, dictionary)
worldnews_words_frequency = frequency_counter(worldnews_words_list, dictionary)

# finding the probability of each word in dictionary given each category P(word_k|category)
p_words_given_hockey = probability_calculator(hockey_words_frequency, hockey_words_list)
p_words_given_movies = probability_calculator(movies_words_frequency, movies_words_list)
p_words_given_nba = probability_calculator(nba_words_frequency, nba_words_list)
p_words_given_news = probability_calculator(news_words_frequency, news_words_list)
p_words_given_nfl = probability_calculator(nfl_words_frequency, nfl_words_list)
p_words_given_politics = probability_calculator(politics_words_frequency, politics_words_list)
p_words_given_soccer = probability_calculator(soccer_words_frequency, soccer_words_list)
p_words_given_worldnews = probability_calculator(worldnews_words_frequency, worldnews_words_list)


# file_maker("validation_output_stem.csv", val_output_list)

# creating the position-list for test-input
temp = []
for i, v in enumerate(validation_input_list):
    temp.append(prediction(v))
    print(i)
file_maker("prediction_for_validation_lamin_10000.csv", temp)




