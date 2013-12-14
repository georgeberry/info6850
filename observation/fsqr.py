'''
@george berry (geb97@cornell.edu) (github.com/georgeberry)
@created: dec 7 2013
'''

#import fuckit
from networkx import *
import datetime as dt
import csv
import cPickle as pickle
import multiprocessing as mp
from functools import wraps
import time
import re
from collections import OrderedDict, Counter, Iterable
import copy
import sys
from itertools import chain
from random import choice
import gc

import numpy as np
import scipy
from sklearn.cluster import KMeans, MiniBatchKMeans
from math import radians, cos, sin, asin, sqrt


##data path

#full data
#main_path = '/Users/georgeberry/Dropbox/INFO6850 project/4sq/from uaz/smaller/FoursquareCheckins20110101-20110731.csv'
#edgelist_path = '/Users/georgeberry/Dropbox/INFO6850 project/4sq/from uaz/smaller/FoursquareFriendship.csv'

#test data
#main_path = '/Users/georgeberry/Dropbox/INFO6850 project/4sq/from uaz/smaller/Fsqrtest.csv'
#edgelist_path = '/Users/georgeberry/Dropbox/INFO6850 project/4sq/from uaz/smaller/FoursquareFriendship.csv'

##global functions

#from: http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 

    # 6367 km is the radius of the Earth
    km = 6367 * c
    return km 



#worker function run by MP process
def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        res = func(*args)
        output.put(res)
        gc.collect()

#how long's it take?
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end - start)
        return result
    return wrapper

def flatten(items, ignore_types=(str, bytes)):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            for element in flatten(x):
                yield element
        else:
            yield x


#this is here because it can't be called from a class
#becuase python MP sucks
@timer
def user_analysis_parallel(users_by_time_during_period, graph, venue_loc, period): #pass info from 1 period
    #errors1 = {} #egos with no friends
    #errors2 = {} #alters with no activity in the previous period
    #based on spot checking and validating with small datset, everything looks good

    config_results = {} #aggregate results for network config
    checkin_by_user_results = {} #observation level results

    for user in users_by_time_during_period['after'].keys():
        try: #can fail if no friends
            ego_net = ego_graph(graph, user)
            #can only influence ego to checkin places never checked in before

            #for node in nodes_iter(ego_net):
            #    ego_net.node[node]['venues'] = []

            temp = users_by_time_during_period['after'][user]

            ego_net.node[user]['new venues'] = [temp[x]['venue'] for x in temp.keys() if temp[x]['venue'] not in users_by_time_during_period['prev_venues'][user]] #holds new venues for period

            ego_net.node[user]['prev venues'] = users_by_time_during_period['prev_venues'][user]


        except:
            #if user not in errors1:
            #    errors1[user] = [['no friends for', user, '; ego =', user]]
            #else:
            #    errors1[user].append(['no previous for user', user, '; ego =', user])                    
            continue


        for vertex in ego_net.nodes(): #venues as node attributes
            if vertex == user:
                continue
            try:
                #we may not want to take a set here: number of exposures is important
                temp = users_by_time_during_period['before'][vertex]
                ego_net.node[vertex]['venues'] = list(set([temp[x]['venue'] for x in temp.keys()]))

            except:
                #if vertex not in errors2:
                #    errors2[vertex] = [['no previous for user', vertex, '; ego =', user]]
                #else:
                #    errors2[vertex].append(['no previous for user', vertex, '; ego =', user])
                continue

        #ego network done
        venues = [y for y in flatten([ego_net.node[x]['venues'] for x in ego_net.nodes()]) if y not in ego_net.node[user]['prev venues']] #all alter venues not in the user's prev venues

        ego_net.node[user]['venues'] = list(set(copy.deepcopy(venues))) #all alter venues #new venues stored in ['new venues']

        venues = Counter(venues)

        for venue in venues.keys():
            #subgraph of only people who have checked into "venue" plus the ego

            sg = ego_net.subgraph([n for n, attributes in ego_net.node.items() if venue in attributes['venues']]) #if venue is in node attrbute 'venues'

            #goodbye, ego
            sg.remove_node(user)

            #graph properties
            configuration = ''.join(str(x) for x in sorted(degree(sg).values()))
            configuration = ':'.join((configuration, str(number_connected_components(sg))))

            num = sg.number_of_nodes()
            edge = sg.number_of_edges()
            if venue in ego_net.node[user]['new venues']:
                ego_checkin = 1
            else:
                ego_checkin = 0

            #store adoption by network configuration
            if configuration not in config_results:
                config_results[configuration] = {'exposures': 0, 'adoptions': 0}

            config_results[configuration]['exposures'] += 1

            if ego_checkin == 1:
                config_results[configuration]['adoptions'] += 1


            #store stats by user, by venue
            if user not in checkin_by_user_results:
                #keyed by user then checkin
                checkin_by_user_results[user] = {venue: {'config': configuration, 'neighbors': num, 'triangles': edge, 'ego checkin': ego_checkin, 'loc':venue_loc[venue], 'period': period} }
            elif venue not in checkin_by_user_results[user]:
                checkin_by_user_results[user][venue] = {'config': configuration, 'neighbors': num, 'triangles': edge, 'ego checkin': ego_checkin, 'loc': venue_loc[venue], 'period': period}

    return config_results, checkin_by_user_results



## classes

class interval: #could be made more efficient with time
    '''
    simply checks whether a date is in the time period
    returns 0 if it's before cutpoint, 1 if it's after
    returns None if date is outside range
    access this check by simply doing instance(date) (__call__ method)
    '''
    def __init__(self, start, cut, end):
        self.start = start
        self.cut = cut
        self.end = end

    def __call__(self, date): #after init
        if self.start <= date < self.cut:
            return 0
        elif self.cut <= date <= self.end: #things on the line are in the subsequent period
            return 1
        else:
            return 2

    def __str__(self):
        return 'start: ' + str(self.start) + '; cut: ' + str(self.cut) + '; end: ' + str(self.end)

    def __repr__(self):
        return __str__(self)


class fsqr:
    '''
    contains data objects and analysis methods
    calling __init__ produces a dict of checkins and a full network
    you probably want to call init and then just call the class with sliding window durations
    __call__ should be configured to accept these, carry out main analysis (MP'd)
    prints to CSV's
    '''
    @timer
    def __init__(self, main_path, edgelist_path): 

        #hack, start mp's now
        #feed the queue later
        #saves on ram
        self.task_queue = mp.Queue()
        self.done_queue = mp.Queue()

        self.NUM_PROC = 6

        #the procs literally just sit around waiting for stuff to go on the queue
        for i in range(self.NUM_PROC):
            mp.Process(target = worker, args = (self.task_queue, self.done_queue)).start()

        self.by_checkin = {}
        self.g = Graph()
        self.venue_loc = {}
        self.user_loc = {}
        self.user_centroid = {}
        self.time_period = {}

        with open(main_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            line_number = 0
            for row in reader:
                user, loc, date, venue = int(row[0]), (float(row[1]), float(row[2])), self.dateify(row[3].strip()), int(row[4])

                self.by_checkin[line_number] = { 'user': user, 'loc': loc, 'date': date, 'venue': venue }
                if venue not in self.venue_loc:
                    self.venue_loc[venue] = loc

                if user not in self.user_loc:
                    self.user_loc[user] = [[loc[0], loc[1]]]
                else:
                    self.user_loc[user].append([loc[0], loc[1]]) 

                line_number += 1

        with open(edgelist_path) as g:
            edgelist = []
            reader = csv.reader(g)
            next(reader)

            for row in reader:
                ego, alter = int(row[0]), int(row[1])
                edgelist.append((ego,alter))

            self.g.add_edges_from(edgelist)

            for node in nodes_iter(self.g):
                self.g.node[node]['venues'] = []

            del edgelist


        #need user centroids to compute distance
        clusters = 3

        for usr in self.user_loc.keys():
            if len(self.user_loc[usr]) >= (clusters + 1) and len(self.user_loc[usr]) <= (45):
                n = np.array(self.user_loc[usr])
                k = KMeans(init='k-means++', n_clusters = clusters, n_init=10).fit(n)
                cent, label = k.cluster_centers_, k.labels_
                mean_group = Counter(list(label)).most_common(1)[0][0] # most common cluster number
                self.user_centroid[usr] = cent[mean_group]
            elif len(self.user_loc[usr]) > 45:
                n = np.array(self.user_loc[usr])
                k = MiniBatchKMeans(init='k-means++', n_clusters = clusters, n_init=10, batch_size = 45).fit(n)
                cent, label = k.cluster_centers_, k.labels_
                mean_group = Counter(list(label)).most_common(1)[0][0] # most common cluster number
                self.user_centroid[usr] = cent[mean_group]   

            else:
                self.user_centroid[usr] = choice(self.user_loc[usr]) #random choice if not enough obs

        del self.user_loc


        #messy 
        #gets first and last dates in data
        #gets difference in days and seconds between these dates
        #assumes we read in by date
        self.time_period = {'start': self.by_checkin[0]['date'], 'stop': self.by_checkin[len(self.by_checkin.items()) - 1]['date'], 'total': self.by_checkin[len(self.by_checkin.items()) - 1]['date'] - self.by_checkin[0]['date'] } 




    @staticmethod #transforms string date into datetime
    def dateify(date_as_string):
        r = [int(x) for x in re.split('\s|:|-', date_as_string)]
        return dt.datetime(r[0], r[1], r[2], r[3], r[4], r[5])


    @staticmethod
    @timer
    def get_venues(by_user): #gets venue information from user information
        by_venue = {}
        for usr in by_user.keys():
            for checkin in by_user[usr]:
                v = by_user[usr][checkin]['venue']
                loc = by_user[usr][checkin]['loc']

                if v not in by_venue:
                    by_venue[v] = {'users': [usr], 'loc': [loc]}
                else:
                    by_venue[v]['users'].append(usr)
                    by_venue[v]['loc'].append(loc)

        for key in by_venue.keys():
            by_venue[key]['loc'] = list(set(by_venue[key]['loc']))
                #if venue not in by_venue:
                #    by_venue[venue] = {'users': [user], 'loc': [loc]}
                #else:
                #    by_venue[venue]['users'].append(user)
                #    by_venue[venue]['loc'].append(loc)

        return by_venue


    def get_users(self, checkin_during_interval): #gets users from checkin dict
        by_user = {}

        for checkin in checkin_during_interval.keys():
            user, loc, date, venue = checkin_during_interval[checkin]['user'], checkin_during_interval[checkin]['loc'], checkin_during_interval[checkin]['date'], checkin_during_interval[checkin]['venue']

            if user not in by_user: #deal with edges later
                by_user[user] = { checkin: { 'loc': loc, 'time': date, 'venue': venue } }

            else: 
                by_user[user][checkin] = { 'loc': loc, 'date': date, 'venue': venue }

        return by_user


    @timer
    def users_by_time(self, durationA, durationB): #builds user_dict
        #need to generate range of intervals based on duration
        #duration is given in number of days

        # |a|b|c|d|e|f|g| # these are days
        #assume we want to analyze a day (durationB) based on the previous two days (durationA)
        #there would have to be an inital offset of durationA
        # there would be (total_days - offset) % durationB analyses
        # if durationA = 2 and durationB = 1, this equals 5
        # if durationA = 1 and durationB = 2, this equals 3
        # we don't double-analyze days in durationB, we jump forward by durationB and drag the durationA window
        # there may be double counting in the durationA window

        if durationA < durationB:
            return 'durationA must be >= durationB'

        self.analysis_periods = []
        self.users_by_time = {}
        #keyed by time period, then by 'before' or 'after' the cutpoint

        days = self.time_period['total'].days
        num_periods = (days - durationA) / durationB
        fd = self.by_checkin.items()[0][1]['date']
        first_day = dt.datetime(fd.year, fd.month, fd.day, 0, 0, 0)

        start = first_day
        cut = first_day + dt.timedelta(durationA)
        end = cut + dt.timedelta(durationB)

        for each in xrange(num_periods):
            self.analysis_periods.append(interval(start, cut, end))
            start += dt.timedelta(durationB)
            cut += dt.timedelta(durationB)
            end += dt.timedelta(durationB)

        #need at each period, for each user, what are the absolute first checkins in durationB for the user in the whole dataset

        period = 0
        prev_venues = {} #previous venues keyed by user
        #use these to check later if there has been a new venue
        #works by adding all the items in durationA to a set object
        #because durationA's will overlap when durationA >= durationB, this works
        #will not be correct if durationA < durationB, so don't do this

        for checkin in self.by_checkin.keys():
            if self.by_checkin[checkin]['user'] not in prev_venues:
                prev_venues[self.by_checkin[checkin]['user']] = set()

        for time_interval in self.analysis_periods:
            before_cut = {}
            after_cut = {}

            for checkin in self.by_checkin.keys():
                checked = self.by_checkin[checkin]

                response = time_interval(checked['date'])
                if response == 0:
                    before_cut[checkin] = checked

                elif response == 1:
                    after_cut[checkin] = checked

                elif response == 2:
                    continue

            #in each period, add the venues in durationA to the prev venues, then attach prev venues to users_by_time

            before_cut = self.get_users(before_cut)

            for user in before_cut.keys():
                if user not in prev_venues:
                    prev_venues[user] = set([before_cut[user][x]['venue'] for x in before_cut[user]])
                    #print prev_venues[user]
                else:
                    #print user, prev_venues[user]
                    temp1 = prev_venues[user] 
                    temp2 = set([before_cut[user][x]['venue'] for x in before_cut[user]])
                    prev_venues[user] = temp1.union(temp2)

            temp_prev = copy.deepcopy(prev_venues) #set objects are weird

            self.users_by_time[period] = {'before': before_cut, 'after': self.get_users(after_cut), 'prev_venues': temp_prev}

            period += 1


    @timer
    def call_parallel(self):

        self.config_results = {}
        self.checkin_by_user_results = {} #dict keyed by user, holds list

        TASKS = [(user_analysis_parallel, (copy.deepcopy(self.users_by_time[period]), copy.deepcopy(self.g), copy.deepcopy(self.venue_loc), copy.deepcopy(period))) for period in self.users_by_time.keys()]

        for task in TASKS:
            self.task_queue.put(task)

        for i in range(len(TASKS)):
            config_results, user_results = self.done_queue.get()
            for config in config_results.keys():
                if config not in self.config_results:
                    self.config_results[config] = config_results[config]
                else:
                    self.config_results[config]['exposures'] = self.config_results[config]['exposures'] + config_results[config]['exposures']
                    self.config_results[config]['adoptions'] = self.config_results[config]['adoptions'] + config_results[config]['adoptions']

            for user in user_results.keys():
                if user not in self.checkin_by_user_results:
                    self.checkin_by_user_results[user] = [user_results[user]]
                else:
                    self.checkin_by_user_results[user].append(user_results[user])

            gc.collect()


        #shut down procs
        for i in range(self.NUM_PROC):
            self.task_queue.put('STOP')



    @timer
    def summary_stats(self):
        #num users in each period
        self.num_users = {k: (len(v['before']), len(v['after'])) for k, v in self.users_by_time.iteritems()}

        self.user_counts_in_period = {}
        self.venue_counts_in_period = {}
        self.user_venue_counts_in_period = {} #keyed by user
        self.total_user_counts = {}
        self.total_venue_counts = {}


        for checkin in self.by_checkin.keys():
            t = self.by_checkin[checkin]

            if t['user'] not in self.total_user_counts:
                self.total_user_counts[t['user']] = 0

            if t['venue'] not in self.total_venue_counts:
                self.total_venue_counts[t['venue']] = 0

            self.total_user_counts[t['user']] += 1
            self.total_venue_counts[t['venue']] += 1


        period = 0

        for time_period in self.analysis_periods:
            self.user_counts_in_period[period] = {}
            self.venue_counts_in_period[period] = {}
            self.user_venue_counts_in_period[period] = {}

            for checkin in self.by_checkin.keys():
                t = self.by_checkin[checkin]
                user, venue, date = t['user'], t['venue'], t['date']

                result = time_period(date)
                if result != 2:
                    if user not in self.user_counts_in_period[period]:
                        self.user_counts_in_period[period][user] = 0
                        self.user_venue_counts_in_period[period][user] = {}

                    if venue not in self.user_venue_counts_in_period[period][user]:
                        self.user_venue_counts_in_period[period][user][venue] =  0

                    if venue not in self.venue_counts_in_period[period]:
                        self.venue_counts_in_period[period][venue] = 0
                    
                    self.user_counts_in_period[period][user] += 1
                    
                    self.user_venue_counts_in_period[period][user][venue] += 1

                    self.venue_counts_in_period[period][venue] += 1

            period += 1


        #other stuff to generate
        #total checkins for each user, for each venue
        #where user has checked in that none of their friends have (in each period)
        #calculate exposures per checkin over time (i.e. reistance)

    @timer
    def output(self, aggregate, full):
        with open(aggregate, 'wb') as w: #aggregate
            writer = csv.writer(w)
            writer.writerow(['network config', 'adoptions', 'exposures', 'convert ratio'])
            
            t = OrderedDict(sorted(self.config_results.items(), key = lambda x: len(x[0])))

            for key in t.keys():
                d = t[key]
                config, adoptions, exposures = key, d['adoptions'], d['exposures']
                writer.writerow([config, adoptions, exposures, float(adoptions)/float(exposures)])

        with open(full, 'wb') as w: #split up
            writer = csv.writer(w)
            writer.writerow(['user', 'venue', 'period', 'neighbors', 'triangles', 'components', 'config', 'long', 'lat', 'km from venue', 'ego checkin', 'ego period checkins', 'ego period checkins at venue', 'venue period checkins', 'user total checkins', 'venue total checkins', 'user centroid lat', 'user centroid long', 'degree', 'clust coeff', 'k core', 'deg cent'])

            deg = self.g.degree()
            clust = clustering(self.g)
            core = core_number(self.g)
            deg_cent = degree_centrality(self.g)

            for key in self.checkin_by_user_results.keys(): #user key
                for chunk in self.checkin_by_user_results[key]: #for each in list of dicts
                    for venue_key in chunk.keys():
                        t = chunk[venue_key]

                        try:
                            uc = self.user_counts_in_period[t['period']][key]
                        except:
                            uc = 0

                        try:
                            uvc = self.user_venue_counts_in_period[t['period']][key][venue_key]
                        except:
                            uvc = 0

                        try:
                            vc = self.venue_counts_in_period[t['period']][venue_key]
                        except:
                            vc = 0

                        user, venue, period, neighbors, triangles, components, config, lon, lat, euc_dist_from_venue, ego_checkin, ego_total_checkins_during_period, ego_checkins_at_venue_during_period, venue_total_checkins_during_period, total_user_counts, tot_ven_counts, centroid_lat, centroid_long, usr_deg, usr_clust, usr_core, usr_deg_cent = key, venue_key, t['period'], t['neighbors'], t['triangles'], t['config'].split(':')[1], t['config'], t['loc'][0], t['loc'][1], haversine(self.venue_loc[venue_key][1], self.venue_loc[venue_key][0], self.user_centroid[key][1], self.user_centroid[key][0]), t['ego checkin'], uc, uvc, vc, self.total_user_counts[key], self.total_venue_counts[venue_key], self.user_centroid[key][1], self.user_centroid[key][0], deg[key], clust[key], core[key], deg_cent[key]



                        writer.writerow([user, venue, period, neighbors, triangles, components, config, lon, lat, euc_dist_from_venue, ego_checkin, ego_total_checkins_during_period, ego_checkins_at_venue_during_period, venue_total_checkins_during_period, total_user_counts, tot_ven_counts, centroid_lat, centroid_long, usr_deg, usr_clust, usr_core, usr_deg_cent])



    def __call__(self, durationA, durationB, output1, output2): #after init
        self.users_by_time(durationA, durationB)
        self.call_parallel()
        self.summary_stats()
        self.output(output1, output2)


