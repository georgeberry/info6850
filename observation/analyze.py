from numpy import *
from pandas import *
import statsmodels.api as sm
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
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels import graphics
import gc

#have users and repeated exposures to venues
#the more i think about it, the more we should log # exposures

filepath1 = '/Users/georgeberry/Documents/info6850/observation/results/monthly_redone_full_1.csv'
filepath2 = '/Users/georgeberry/Documents/info6850/observation/results/monthly_redone_full_2.csv'
filepath3 = '/Users/georgeberry/Documents/month_by_month_1checkin_egos_1st_time.csv'


## FUNCTIONS ##

def clean(path):
    df = read_csv(path, index_col = [0,1]) #usecols = [0,1,2,3,4,5,7,8,9,10])

    #drop larger ego networks
    #df = df[df.neighbors <= 5]

    #generate dummy variables for configs
    #for elem in df['config'].unique():
    #    #if elem != '0:1':
    #    df['config ' + str(elem)] = df['config'] == elem

    #for elem in df['period'].unique():
    #    #if elem != 0:
    #    df['period ' + str(elem)] = df['period'] == elem

    print 'dummies made'

    #df = df.drop('config',1)
    #df = df.drop('period',1)

    #df = df.astype(float)

    to_drop = []

    #for col in xrange(len(df.columns)):
    #    colname = df.columns[col]
    #    if col > 17 and col < 68 and  np.sum(df.ix[0:,col]) < 1000:
    #        #df = df[df[colname] != 1] #gets rid of checkin observations for these rare cases
    #        to_drop.append(df.columns[col])

    #df = df[df['config 0:1'] != 1]
    #df = df.drop('config 0:1', 1)

    print to_drop

    #for each in to_drop:
    #    df = df.drop(each, 1)

    return df

def check_corr(frame):
    for column in frame.columns:
        for other_column in frame.columns:
            corr = stats.pearsonr(frame[column], frame[other_column])
            if corr[0] > .8 and column != other_column:
                print column, other_column, stats.pearsonr(frame[column], frame[other_column])

def delete_spaces(frame):
    new_names = []
    for each in frame.columns:
        new_names.append(each.replace(' ', ''))

    frame.columns = new_names
    return frame

def do_neg_bn(y, X):
    glm_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    res = glm_nb.fit()
    print res.summary()
    nobs = res.nobs
    yhat = res.mu

    to_drop = []
    for each in xrange(len(res.mu)):
        if res.mu[each] > 100:
            to_drop.append(each)
            print 'row index:', each, ', value', res.mu[each]

    plt.figure()
    plt.scatter(yhat, y)
    line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=False)).fit().params
    fit = lambda x: line_fit[1] + line_fit[0] * x
    plt.plot(np.linspace(0, 1, nobs), fit(np.linspace(0, 1, nobs)))
    plt.title('Model Fit Plot')
    plt.ylabel('Observed values')
    plt.xlabel('Fitted values')
    plt.show()

    plt.figure()
    plt.scatter(yhat, res.resid_pearson)
    plt.plot([0.0, 1.0], [0.0, 0.0], 'k-')
    plt.title('Residual Dependence Plot')
    plt.ylabel('Pearson Residuals')
    plt.xlabel('Fitted values')
    plt.show()

    n, bins, patches = plt.hist(y_reg, 500, normed=1, facecolor='g', alpha=0.75)
    plt.show()

    graphics.gofplots.qqplot(res.mu, line='r')

    return to_drop

def condense(frame, index_num):
    arr = {d: [] for d in frame.columns}
    arr['counts'] = []

    for name, group in frame.groupby(level=index_num):
        means = group.mean()
        sums = group.sum()
        mins = group.min()
        maxs = group.max()
        L = len(group)
        #need to put them together

        for column in xrange(len(frame.columns)): #sum of 0-1 #columns, mean of others
            if frame.columns[column] in ['ego period checkins', 'ego period checkins at venue', 'venue period checkins', 'venue total checkins', 'user centroid lat', 'user centroid long', 'degree', 'clust coeff', 'k core', 'deg cent']:
                arr[frame.columns[column]].append(sums[column]/L)
            else:
                arr[frame.columns[column]].append(sums[column])
        arr['counts'].append(L)

    return DataFrame(arr)


### BY USER DATA ###

#generate dummy variables for configs

df = clean(filepath3)

df_user = condense(df, 0)

x_reg = delete_spaces(df_user)

x_reg.to_csv('/Users/georgeberry/Documents/wei_out.csv')

y_reg = copy.deepcopy(df_user.loc[0:,'egocheckin'])

#put stuff in regression here
x_reg = df_user.loc[0:, ['triangles','counts','kmfromvenue', 'components', 'config00:2', 'config11:1', 'neighbors']]

to_drop = do_neg_bn(y_reg, x_reg)

#prune and repeat

x_reg = x_reg.drop(x_reg.index[[to_drop]])
y_reg = y_reg.drop(y_reg.index[[to_drop]])

to_drop = do_neg_bn(y_reg, x_reg)


### BY VENUE DATA ###

df = clean(filepath1)

df_venue = condense(df, 1)

x_reg = delete_spaces(df_venue)

x_reg.to_csv('/Users/georgeberry/Documents/venue_2_out.csv')

y_reg = copy.deepcopy(df_venue.loc[0:,'ego checkin'])
x_reg = df_venue.loc[0:, ['triangles','counts','kmfromvenue', 'components', 'config00:2', 'config11:1', 'neighbors']]

to_drop = do_neg_bn(y_reg, x_reg)

x_reg = x_reg.drop(x_reg.index[[to_drop]])
y_reg = y_reg.drop(y_reg.index[[to_drop]])

to_drop = do_neg_bn(y_reg, x_reg)


### FULL DATA ###

## decision tree learning ##

# random decision forest
# bootstrapping (sample w/o replacement) procedure
# going to use information entropy && gini impurity and see what works better

del df

df1 = delete_spaces(clean(filepath1))
df2 = delete_spaces(clean(filepath2))

y_base = copy.deepcopy(df1.loc[0:,'egocheckin'])
y_base = y_base.replace(0, -1)

x_base = df1.loc[0:, ['triangles', 'components']]

train_rows = random.sample(x_base.index, 100000)
y_train = y_base.ix[train_rows]
x_train = x_base.ix[train_rows]

test_rows = random.sample(x_base.index, 100000)
y_test = y_base.ix[test_rows]
x_test = x_base.ix[test_rows]

# tree #

clf = tree.DecisionTreeClassifier(criterion='entropy',min_samples_split = 500, min_samples_leaf = 250, max_depth = 5)
clf = clf.fit(x_train, y_train)

print clf.score(x_test, y_test)

dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("/Users/georgeberry/Desktop/iris.pdf") 

# forest #

clf = RandomForestClassifier(n_estimators = 10, criterion='entropy', min_samples_split = 100, min_samples_leaf = 100, n_jobs = 6).fit(x_base, y_base)

print clf.score(x_test, y_test)

for each in izip(clf.feature_importances_, x_train.columns):
    print each

#predict on other dataset

y_base2 = copy.deepcopy(df2.loc[0:,'egocheckin'])
y_base2 = y_base2.replace(0, -1)

x_base2 = df2.loc[0:, ['triangles','kmfromvenue', 'components', 'config00:2', 'config11:1', 'config0:1']]


print clf.score(x_base2, y_base2)