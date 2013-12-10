'''
@george berry (geb97@cornell.edu) (github.com/georgeberry)
@created: dec 7 2013
'''

import fuckit
from networkx import *
import datetime as dt
import csv
import cPickle as pickle
from functools import wraps
import time
import re
from collections import OrderedDict, Counter, Iterable
import copy
import sys
from itertools import chain


#global functions

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end - start)
        return result
    return wrapper


def flatten(items, ignore_types=(str, bytes)): #yeah classmethod generator
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            for element in flatten(x):
                yield element
        else:
            yield x


main_path = '/Users/georgeberry/Dropbox/INFO6850 project/4sq/from uaz/smaller/FoursquareCheckins20110101-20110731.csv'
edgelist_path = '/Users/georgeberry/Dropbox/INFO6850 project/4sq/from uaz/smaller/FoursquareFriendship.csv'

main_path = '/Users/georgeberry/Dropbox/INFO6850 project/4sq/from uaz/smaller/Fsqrtest.csv'
edgelist_path = '/Users/georgeberry/Dropbox/INFO6850 project/4sq/from uaz/smaller/FoursquareFriendship.csv'

output1 = 'summary.csv'
output2 = 'big_file.csv'

class interval:
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
        if self.start < date < self.cut:
            return 0
        elif self.cut < date < self.end:
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
    this will return a class object with the appropriate objects
    calling __init__ produces a dict of checkins and a full network
    '''
    @timer
    def __init__(self, main_path, edgelist_path): 
        self.by_checkin = {}
        self.g = Graph()
        self.venue_loc = {}

        with open(main_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            line_number = 0
            for row in reader:
                user, loc, date, venue = int(row[0]), (float(row[1]), float(row[2])), self.dateify(row[3]), int(row[4])

                self.by_checkin[line_number] = { 'user': user, 'loc': loc, 'date': date, 'venue': venue }
                if venue not in self.venue_loc:
                    self.venue_loc[venue] = loc

                line_number += 1

        #by_user, get_rid_of = cls.chuck_em(by_user) #gets rid of the junk


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

        #self.by_checkin = OrderedDict(sorted(self.by_checkin.items(), key=lambda x: (x[1]['date'] - dt.datetime(1970,1,1)).total_seconds())) #sorts dict by date

        #messy 
        #gets first and last dates in data
        #gets difference in days and seconds between these dates
        self.time_period = {'start': self.by_checkin.items()[0][1]['date'], 'stop': self.by_checkin.items()[len(self.by_checkin.items()) - 1][1]['date'], 'total': self.by_checkin.items()[len(self.by_checkin.items()) - 1][1]['date'] - self.by_checkin.items()[0][1]['date'] } 

    #@staticmethod
    #@timer
    #def chuck_em(by_user): #returns list of users with no friends
    #    get_rid_of = []
    #    for usr in by_user.keys():
    #        if len(by_user[usr]['neighbors']) < 2:
    #            del by_user[usr]
    #            get_rid_of.append(usr)
    #    return by_user, get_rid_of

    @staticmethod #quickly transforms string date into datetime
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
            return 'durationA msut be >= durationB'

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
    def user_analysis(self): #takes one time period, composed of before and after
        #don't be scared by errors, they can arise in many cases
        #ego has no friends over the dataset
        #alter has no activity in the previous period
        
        errors1 = {} #egos with no friends
        errors2 = {} #alters with no activity in the previous period
        #based on spot checking and validating with small datset, everything looks good

        self.config_results = {}
        self.checkin_by_user_results = {}



        for period in self.users_by_time.keys():

            for user in self.users_by_time[period]['after'].keys():
                try: #can fail if no friends
                    ego_net = ego_graph(self.g, user)
                    #can only influence ego to checkin places never checked in before

                    #for node in nodes_iter(ego_net):
                    #    ego_net.node[node]['venues'] = []

                    temp = self.users_by_time[period]['after'][user]

                    ego_net.node[user]['new venues'] = [temp[x]['venue'] for x in temp.keys() if temp[x]['venue'] not in self.users_by_time[period]['prev_venues'][user]] #holds new venues for period


                except:
                    if user not in errors1:
                        errors1[user] = [['no friends for', user, 'in time', period, '; ego =', user]]
                    else:
                        errors1[user].append(['no previous for user', user, 'in time', period, '; ego =', user])                    
                    continue


                for vertex in ego_net.nodes(): #venues as node attributes
                    if vertex == user:
                        continue
                    try:
                        #we may not want to take a set here: number of exposures is important
                        temp = self.users_by_time[period]['before'][vertex]
                        ego_net.node[vertex]['venues'] = list(set([temp[x]['venue'] for x in temp.keys()]))

                    except:
                        if vertex not in errors2:
                            errors2[vertex] = [['no previous for user', vertex, 'in time', period, '; ego =', user]]
                        else:
                            errors2[vertex].append(['no previous for user', vertex, 'in time', period, '; ego =', user])
                        continue

                #ego network done
                venues = [y for y in flatten([ego_net.node[x]['venues'] for x in ego_net.nodes()])] #should get counts
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
                    if configuration not in self.config_results:
                        self.config_results[configuration] = {'exposures': 0, 'adoptions': 0}

                    self.config_results[configuration]['exposures'] += 1

                    if ego_checkin == 1:
                        self.config_results[configuration]['adoptions'] += 1


                    #store stats by user, by venue
                    if user not in self.checkin_by_user_results:
                        #keyed by user then checkin
                        self.checkin_by_user_results[user] = {venue: {'config': configuration, 'neighbors': num, 'triangles': edge, 'ego checkin': ego_checkin, 'loc': self.venue_loc[venue], 'period': period} }
                    elif venue not in self.checkin_by_user_results[user]:
                        self.checkin_by_user_results[user][venue] = {'config': configuration, 'neighbors': num, 'triangles': edge, 'ego checkin': ego_checkin, 'loc': self.venue_loc[venue], 'period': period}

            print 'period', period, 'done'
        return self.config_results, self.checkin_by_user_results

    @timer
    def output(self):
        with open(output1, 'wb') as w: #aggregate
            writer = csv.writer(w)
            for key in self.config_results.keys():
                t = self.config_results[key]
                config, adoptions, exposures = key, t['adoptions'], t['exposures']
                writer.writerow([config, adoptions, exposures, float(adoptions)/float(exposures)])

        with open(output2, 'wb') as w: #split up
            writer = csv.writer(w)
            for key in self.checkin_by_user_results.keys():
                for venue_key in self.checkin_by_user_results[key].keys():
                    t = self.checkin_by_user_results[key][venue_key]

                    user, venue, period, neighbors, triangles, long, lat, config, ego_checkin = key, venue_key, t['period'], t['neighbors'], t['triangles'], t['loc'][0], t['loc'][1], t['config'], t['ego checkin']

                    writer.writerow([user, venue, period, neighbors, triangles, loc, config, ego_checkin])


    @timer
    def summary_stats(self):
        #num users in each period
        self.num_users = {k: (len(v['before']), len(v['after'])) for k, v in self.users_by_time.iteritems()}


    def __call__(self, durationA, durationB): #after init
        self.users_by_time(durationA, durationB)
        self.user_analysis()
        self.summary_stats()





