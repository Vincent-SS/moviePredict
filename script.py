import numpy as np
import pandas as pd
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

train_data = pd.read_csv(sys.argv[1])
valid_data = pd.read_csv(sys.argv[2])

train_data = train_data[['movie_id', 'cast', 'crew', 'budget', 'genres', 'homepage', 'original_language', 'production_companies', 'runtime', 'revenue', 'rating']]
valid_data = valid_data[['movie_id', 'cast', 'crew', 'budget', 'genres', 'homepage', 'original_language', 'production_companies', 'runtime', 'revenue', 'rating']]

# Process homepage
train_data.loc[train_data.homepage != '', "homepage"] = np.int64(1)
train_data.loc[train_data.homepage != 1, "homepage"] = np.int64(0)
valid_data.loc[valid_data.homepage != '', "homepage"] = np.int64(1)
valid_data.loc[valid_data.homepage != 1, "homepage"] = np.int64(0)

# Process train_data
for columnss in ['cast', 'genres', 'production_companies']:
    i = 0
    for each_row in train_data[columnss]:
        ec = eval(each_row)
        results = []
        for r in ec:
            results.append(r['id'])
        train_data[columnss][i] = results
        i+=1        

# Process crew
i = 0
for each_row in train_data['crew']:
    ec = eval(each_row)
    results = []
    for r in ec:
        if (r['job'] == 'Director'):
            results.append(r['id'])
    train_data['crew'][i] = results
    i+=1

# Process valid_data
for columnss in ['cast', 'genres', 'production_companies']:
    i = 0
    for each_row in valid_data[columnss]:
        ec = eval(each_row)
        results = []
        for r in ec:
            results.append(r['id'])
        valid_data[columnss][i] = results
        i+=1       

# Process crew
i = 0
for each_row in valid_data['crew']:
    ec = eval(each_row)
    results = []
    for r in ec:
        if (r['job'] == 'Director'):
            results.append(r['id'])
    valid_data['crew'][i] = results
    i+=1

# Generate final columns for comparison
# preprocessing the data, to get TOP16 occurrences for each row
compare_columns = []
for column in ['cast', 'crew', 'genres', 'production_companies']:
    c_value = train_data[column].values
    c_count = {}
    for each_row_c in c_value:
        for each_c in each_row_c:
            curr = c_count.get(each_c)
            if curr is None:
                c_count[each_c] = 1
            else:
                c_count[each_c]+=1
    top_c = sorted(c_count.items(), key=lambda x:x[1], reverse=True)
    top = 16
    top_c = top_c[:top]
    top_c_columns = []
    # create columns and extend to compare_columns
    for i in range(top):
        new_col = "top_"+column+"_"+str(i+1)
        top_c_columns.append(new_col)
    compare_columns.extend(top_c_columns)
    # Convert value to 1 or 0 (exist 1; else 0)
    for i in range(top):
        train_data[top_c_columns[i]] = train_data[column].apply(lambda x: 1 if top_c[i][0] in x else 0)
        valid_data[top_c_columns[i]] = valid_data[column].apply(lambda x: 1 if top_c[i][0] in x else 0)

# Add runtime and budget
compare_columns.append('runtime')
compare_columns.append('budget')

# Part-1 Regression
out_id = valid_data.movie_id.values
Xtrain = train_data[compare_columns]
Xtest = valid_data[compare_columns]
Ytrain = train_data['revenue']
Ytest = valid_data['revenue']

rf = RandomForestRegressor(n_estimators=150)
rf.fit(Xtrain, Ytrain)
Ypred = rf.predict(Xtest)
mse = mean_squared_error(Ytest, Ypred)
corr = np.corrcoef(Xtrain, Ytrain, rowvar=False)[-1, :-1]
corr = corr.max()
f = open("PART1.summary.csv", "w")
f.write("MSE,correlation\n")
f.write("{},{:.2f}\n".format(mse, corr))
f.close()

f = open("PART1.output.csv", "w")
f.write("movie_id,predicted_revenue\n")
for i in range(Ypred.shape[0]):
    f.write("{},{}\n".format(out_id[i], Ypred[i]))
f.close()

# Part-2 Classification
out_id = valid_data.movie_id.values
Xtrain_2 = train_data[compare_columns]
Xtest_2 = valid_data[compare_columns]
Ytrain_2 = train_data['rating']
Ytest_2 = valid_data['rating']

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xtrain_2, Ytrain_2)
Ypred_2 = knn.predict(Xtest_2)
pre = precision_score(Ytest_2, Ypred_2, average='macro')
recall = recall_score(Ytest_2, Ypred_2, average='macro')
acc = accuracy_score(Ytest_2, Ypred_2)

f = open("PART2.summary.csv", "w")
f.write("average_precision,average_recall,accuracy\n")
f.write("{:.2f},{:.2f},{}\n".format(pre, recall, acc))
f.close()

f = open("PART2.output.csv", "w")
f.write("movie_id,predicted_rating\n")
for i in range(Ypred_2.shape[0]):
    f.write("{},{}\n".format(out_id[i], Ypred_2[i]))
f.close()
