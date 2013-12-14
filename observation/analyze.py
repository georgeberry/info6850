from numpy import *
from pandas import *
import statsmodels.formula.api as smf
from statsmodels.iolib import table as t
from statsmodels.iolib.summary2 import summary_col
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import svm, tree
import random
from itertools import izip
from sklearn.externals.six import StringIO
import pydot

#have users and repeated exposures to venues
#the more i think about it, the more we should log # exposures

filepath = '/Users/georgeberry/Dropbox/INFO6850 project/4sq/uaz data/results/monthly_redone_full_1.csv'

#want users and locations to be our two index columns

df = read_csv(filepath, index_col = [0,1]) #usecols = [0,1,2,3,4,5,7,8,9,10])


## set up dummies with the beautiful pandas ##

'''df_subset = df.loc[0:,'neighbors':'euc dist from venue']
df_subset = df_subset.drop('config',1)

df_mean_centered = DataFrame()

for name, group in df_subset.groupby(level=0):
    df_mean_centered = concat([df_mean_centered, (group - group.mean())], axis = 0)

print 'mean centered'

df = concat([df_mean_centered, df.loc[0:,['config', 'period', 'ego checkin']]],axis=1)

df.index = df_subset.index

print 'stitched together'
'''

#drop larger ego networks
df = df[df.neighbors <= 5]

#generate dummy variables for configs
for elem in df['config'].unique():
    #if elem != '0:1':
    df['config ' + str(elem)] = df['config'] == elem

for elem in df['period'].unique():
    #if elem != 0:
    df['period ' + str(elem)] = df['period'] == elem


print 'dummies made'

df = df.drop('config',1)
df = df.drop('period',1)

df = df.astype(float)


'''
## traditional soc sci statistics ##

df_endog = copy.deepcopy(df.loc[0:,'ego checkin'])

df_log = df.drop('ego checkin',1)

cols = df.columns
#drop all configs with fewer than 100 observations
for col in xrange(len(sum)):
    to_drop = []
    if col > 5 and sum[col] < 100:
        to_drop.append(cols[col])
    for each in to_drop:
        df = df.drop(each, 1)

df_exog = df_log

print 'model starting'

logit_mod = smf.Logit(df_endog, df_exog)
logit_res = logit_mod.fit()
logit_margeff = logit_res.get_margeff(method='dydx', at='overall')
print logit_margeff.summary()

print 'done' 
'''


## machine learning ##

# random decision forest
# bootstrapping (sample w/o replacement) procedure
# going to use information entropy && gini impurity and see what works better

#drop all configs with fewer than 100 observations
to_drop = []

for col in xrange(len(df.columns)):
    colname = df.columns[col]
    if col > 11 and np.sum(df.ix[0:,col]) < 100:
        df = df[df[colname] != 1]
        to_drop.append(df.columns[col])

df = df[df['config 0:1'] != 1]
df = df.drop('config 0:1', 1)
df = df.drop('user total checkins', 1)

print to_drop

for each in to_drop:
    df = df.drop(each, 1)

y_base = copy.deepcopy(df.loc[0:,'ego checkin'])
y_base = np.where(y_base == 1, y_base, -1)

x_base = df.drop('ego checkin',1)
x_base = x_base.drop('ego period checkins at venue',1)

train_rows = random.sample(df.index, 10000)
y_train = y_base.ix[train_rows]
x_train = x_base.ix[train_rows]

test_rows = random.sample(df.index, 10)
y_test = y_base.ix[test_rows]
x_test = x_base.ix[test_rows]

# tree #

clf = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split = 100, min_samples_leaf = 50)
clf = clf.fit(x_train, y_train)

dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("/Users/georgeberry/Desktop/iris.pdf") 

# forest #

clf = RandomForestClassifier(n_estimators = 50, criterion='entropy', max_depth = 10, min_samples_split = 100, min_samples_leaf = 50, n_jobs = 6).fit(x_train, y_train)

print clf.score(x_test, y_test)

for each in izip(clf.feature_importances_, x_train.columns):
    print each

# svm

#sclf = svm.SVC(kernel='sigmoid')
#print sclf.fit(x_train,y_train)
#print np.sum(sclf.predict(x_test) - np.array(y_train))
#print np.linalg.norm(np.sum(sclf.predict(x_test) - np.array(y_train)))/len(y_train)

sclf = svm.SVC(kernel='linear')
print sclf.fit(x_train,y_train)
print np.sum(sclf.predict(x_test) - np.array(y_train))
print np.linalg.norm(np.sum(sclf.predict(x_test) - np.array(y_train)))/len(y_train)

#sclf = svm.SVC()
#print sclf.fit(x_train,y_train)
#print np.sum(sclf.predict(x_test) - np.array(y_train))
#print np.linalg.norm(np.sum(sclf.predict(x_test) - np.array(y_train)))/len(y_train)
