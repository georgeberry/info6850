load root files (csv):
    - checkin file with (5): user_id, lat, long, time (m/d/y time), location_id
    - we have a friendlist with (2): user_id_1, user_id_2
    - hometown with (2): user_id and hometown_id

variables:
    - time period (both before and after)
    - distance from checkin (or other user-centroid measure)

implementation: 
    - given time period A and time period B, find out:
        -- for each ego network, the checkins of friends at time A
        -- for each ego network, the checkins of ego at time B
        -- for each venue checked into by 1+ friend(s) in the ego network
            --- network structure of friends who checked in
            --- counts of total of that condition for each user
            --- ratio of open to closed triads
            --- number of components
            --- whether ego checked in in time B
        --- count only first time checkins for ego in time period B as result of influence


    - given a venue, find out:
        -- number of checkins
        -- summary network statistics about users checking in
            --- components, clustering, degree
        -- geographic information

    - given a user, find out:
        -- number of checkins
        -- total ego network summary statistics
            --- degree, clustering, number of places checked into by all
        -- summary geolocation statistics
            --- centroid, cluster analysis, centroid of largest cluster, variance, variance of largest cluster

    - exclude users from data (prune) based on:
        -- no friends

    


data structure:
#checkins have users, location, and time
by_checkin = { 
    checkin_num : { user: 'string', geo  = (lat, long), time = datetime.object, venue = venue} }


#users have checkins and edges
#checkins have location and time
by_user = { 
    user : { 
        checkin_num : { loc = (lat, long), time = datetime.object, venue = venue } 
    } 
}

by_venue = { venue: [users], set((long,lat))}

#when creating NX object
#add attributes to each node venues above
graph['venues'] = [venues]
#all venues added for alters
#ONLY new venues added for egos
#allows easy matching
#['new venues'] for the ego



classes and functions:

    - fsqr class
        -- call fsqr(main_path, edgelist_path) to setup
        -- call class.

    - interval class
        -- holds two time periods, lets you know if a date is in either or neither


stuff generated in kleinberg paper:
    community level
    - Number of members (|C|).
    - Number of individuals with a friend in C (the fringe of C) .
    - Number of edges with one end in the community and the other in the fringe.
    - Number of edges with both ends in the community, |EC |. Thenumberofopentriads:|{(u,v,w)|(u,v)∈EC ∧(v,w)∈EC ∧(u,w)∈/EC ∧u̸=w}|. The number of closed triads: |{(u,v,w)|(u,v) ∈ EC ∧(v,w) ∈ EC ∧(u,w) ∈ EC}|.
    - The ratio of closed to open triads.
    - The fraction of individuals in the fringe with at least k friends in the community for 2 ≤ k ≤ 19. The number of posts and responses made by members of the community.
    - The number of members of the community with at least one post or response.
    - The number of responses per post.
    
    user level
    - Number of friends in community (|S|).
    - Number of adjacent pairs in S (|{(u, v)|u, v ∈ S ∧ (u, v) ∈ EC }|).
    - Number of pairs in S connected via a path in EC .
    - Average distance between friends connected via a path in EC .
    - Number of community members reachable from S using edges in EC . Average distance from S to reachable community members using edges in EC . The number of posts and response made by individuals in S.
    - The number of individuals in S with at least 1 post or response.