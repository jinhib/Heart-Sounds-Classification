import numpy as np
import math
import csv


def bhattacharyya_distance(hist1,  hist2):
    # calculate mean of hist1 and hist2
    size_of_hist1 = len(hist1)
    size_of_hist2 = len(hist2)
    h1_ = np.mean(hist1)
    h2_ = np.mean(hist2)

    # calculate score
    score = 0
    for i in range(size_of_hist1):
        score += math.sqrt(hist1[i] * hist2[i])

    if score == 0:
      return 0
    # print h1_,h2_,score;
    score = math.sqrt(1 - (1 / math.sqrt(h1_ * h2_ * size_of_hist1 * size_of_hist2)) * score)

    return score


file_path = './'
file_name = 'img_features_86.73.csv'
img_path = file_path + file_name

dataset = np.genfromtxt(img_path, delimiter=',', encoding='UTF8')
dataset = dataset[1:]
print('len_of_original_features : ', len(dataset[0]) - 1)

set_1 = []
set_2 = []
for data in dataset:
  if data[-1] == 0:
    set_1.append(data)
  else:
    set_2.append(data)

set_1 = np.array(set_1)
set_2 = np.array(set_2)

scores = []
for index in range(len(dataset[0]) - 1):
  score = bhattacharyya_distance(set_1[:, index], set_2[:, index])
  temp = [score, index]
  scores.append(temp)


sorted_list = sorted(scores, key=lambda x: x[0], reverse=True)
sorted_list = np.array(sorted_list)
bhattacharyya_rank_index = sorted_list[:, 1]

feature_name = np.genfromtxt(img_path, delimiter=',', encoding='UTF8', dtype=str)
feature_name = feature_name[0]

original_dataset = np.genfromtxt(img_path, delimiter=',', encoding='UTF8', skip_header=1)
labels = original_dataset[:, -1]
labels = np.expand_dims(labels, axis=1)
dataset = original_dataset[:, :-1]
dataset = np.transpose(dataset)
feature_name = np.transpose(feature_name)

subset = []
f_name = []
for index in bhattacharyya_rank_index[:10]:
  subset.append(dataset[int(index)])
  f_name.append(feature_name[int(index)])

f_name.append('AA')
print('len_of_bhattacharyya_features : ', len(subset))

subset = np.transpose(subset)
f_name = np.transpose(f_name)
f_name = np.expand_dims(f_name, axis=0)

dataset = np.concatenate((subset, labels), axis=1)
bhattacharyya_dataset = np.concatenate((f_name, dataset), axis=0)

f = open('bhattacharyya_' + str(len(subset[0])) + '_' + file_name, 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
for line in bhattacharyya_dataset:
  wr.writerow(line)
f.close()
