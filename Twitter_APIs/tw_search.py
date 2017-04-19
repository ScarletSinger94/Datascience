#!/usr/bin/env python
# TWITTER SEARCH CRAWLER
#
# By: Ziad Matni
# Last modified: 10/29/2013
# -*- coding: utf-8 -*-

import time, datetime, sys
import re, math, argparse
#import urllib, urlparse
from urllib.parse import urlparse #Dongho 03/28/16
import csv
import tweepy
#import simplejson
#import MySQLdb

# URL CLEANUP
def url_fix(s, charset='utf-8'):
    if isinstance(s, unicode):
        s = s.encode(charset, 'ignore')
    scheme, netloc, path, qs, anchor = urlparse.urlsplit(s)
    path = urllib.quote(path, '/%')
    qs = urllib.quote_plus(qs, ':&=')
    return urlparse.urlunsplit((scheme, netloc, path, qs, anchor))

# COMMAND PARSER
def tw_parser():
    global qw, ge, l, t, c, d

# USE EXAMPLES:
# =-=-=-=-=-=-=
# % twsearch <search term>              --- searches term
# % twsearch <search term> -g sf      --- searches term in SF geographic box <DEFAULT = none>
# % twsearch <search term> -l en      --- searches term with lang=en (English) <DEFAULT = en>
# % twsearch <search term> -t {m,r,p} --- searches term of type: mixed, recent, or popular <DEFAULT = recent>
# % twsearch <search term> -c 12      --- searches term and returns 12 tweets (count=12) <DEFAULT = 1>
# % twsearch <search term> -o {ca, tx, id, co, rtc)   --- searches term and sets output options <DEFAULT = ca, tx>

# Parse the command
    parser = argparse.ArgumentParser(description='Twitter Search')
    parser.add_argument(action='store', dest='query', help='Search term string')
    parser.add_argument('-g', action='store', dest='loca', help='Location (lo, nyl, nym, nyu, dc, sf, nb')
    parser.add_argument('-l', action='store', dest='l', help='Language (en = English, fr = French, etc...)')
    parser.add_argument('-t', action='store', dest='t', help='Search type: mixed, recent, or popular')
    parser.add_argument('-c', action='store', dest='c', help='Tweet count (must be <50)')
    parser.add_argument('-d', action='store', dest='d', help='Destination for output: either screen or DB')
    parser.add_argument('-o', action='store', dest='out', nargs='+', help='Output options: ca, tx, id, co, rtc')
    args = parser.parse_args()

    qw = args.query     # Actual query word(s)
    ge = ''

    # Location
    loca = args.loca
    if (not(loca in ('lo', 'nyl', 'nym', 'nyu', 'dc', 'sf', 'nb')) and (loca)):
        print ("ERROR: Location must be one of these: lo, nyl, nym, nyu, dc, sf, nb")
        exit()
    if loca:
        ge = locords[loca]

    # Language
    l = args.l
    if (not l):
        l = "en"
    if (not(l in ('en','fr','es','po','ko', 'ar'))):
        print ("ERROR: Languages currently supported are: en (English), fr (French), es (Spanish), po (Portuguese), ko (Korean), ar (Arabic)")
        exit()

    # Tweet type
    t = args.t
    if (not t):
        t = "recent"
    if (not(t in ('mixed','recent','popular'))):
        print ("ERROR: Search type must be one of: (m)ixed, (r)ecent, or (p)opular")
        exit()

    # Tweet count
    if args.c:
        c = int(args.c)
        if (c > cmax):
            print ("Resetting count to ",cmax," (maximum allowed)")
            c = cmax
        if (not (c) or (c < 1)):
            c = 1
    if not(args.c):
        c = 1

    # Output destination
    d = args.d
    if (not(d in ('sc', 'db')) and (not d)):
        print ("ERROR: You didn't specify output destination: either sc (screen) or db (MySQL database). Defaulting to sc (screen)")
        d = "sc"

    # Output choices --- maybe remove?
    choices = ['ca', 'tx', 'id', 'co', 'rtc']
    out = args.out
    if (not(out) or (not((set(out) & set(choices))))):
        print ("ERROR: Output choices must be one of: ca, tx, id, co, rtc. Using defaults.")
        out = ["ca","tx"]
    else:
        out = (set(args.out) & set(choices))

    print
#    print "Q: %s, LOC: %s, LANG: %s, TYPE: %s, COUNT: %s, D: %s, OUT: %s" %(qw ,ge ,l ,t ,c, d, out)

# AUTHENTICATION (OAuth)
def tw_oauth(authfile):
    with open(authfile, "r") as f:
        ak = f.readlines()
    f.close()
    auth1 = tweepy.auth.OAuthHandler(ak[0].replace("\n",""), ak[1].replace("\n",""))
    auth1.set_access_token(ak[2].replace("\n",""), ak[3].replace("\n",""))
    return tweepy.API(auth1)

def tw_search_json(query, cnt=5):
    authfile = './auth.k'
    api = tw_oauth(authfile)
    results = {}
    meta = {
        'username': 'text',
        'usersince': 'date',
        'followers': 'numeric',
        'friends': 'numeric',
        'authorid': 'text',
        'authorloc': 'geo',
        'geoenable': 'boolean',
        'source': 'text'
    }
    data = []
    for tweet in tweepy.Cursor(api.search, q=query, count=cnt).items():
        dTwt = {}
        dTwt['username'] = tweet.author.name
        dTwt['usersince'] = tweet.author.created_at      #author/user profile creation date
        dTwt['followers'] = tweet.author.followers_count #number of author/user followers (inlink)
        dTwt['friends']   = tweet.author.friends_count   #number of author/user friends (outlink)
        dTwt['authorid']  = tweet.author.id              #author/user ID#
        dTwt['authorloc'] = tweet.author.location        #author/user location
        dTwt['geoenable'] = tweet.author.geo_enabled     #is author/user account geo enabled?
        dTwt['source']    = tweet.source                 #platform source for tweet
        data.append(dTwt)
    results['meta'] = meta
    results['data'] = data
    return results


# TWEEPY SEARCH FUNCTION
def tw_search(api):
    counter = 0
    # Open/Create a file to append data
    csvFile = open('result.csv','w')
    #Use csv Writer
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(["created", "text", "retwc", "hashtag", "followers", "friends"])
	
    for tweet in tweepy.Cursor(api.search,
                                q = qw,
                                g = ge,
                                lang = l,
                                result_type = t,
                                count = c).items():

        #TWEET INFO
        created = tweet.created_at   #tweet created
        text    = tweet.text         #tweet text
        tweet_id = tweet.id          #tweet ID# (not author ID#)
        cords   = tweet.coordinates  #geographic co-ordinates
        retwc   = tweet.retweet_count #re-tweet count
        try:
            hashtag = tweet.entities[u'hashtags'][0][u'text'] #hashtags used
        except:
            hashtag = "None"
        try:
            rawurl = tweet.entities[u'urls'][0][u'url'] #URLs used
            urls = url_fix(rawurl)
        except:
            urls    = "None"
        #AUTHOR INFO
        username  = tweet.author.name            #author/user name
        usersince = tweet.author.created_at      #author/user profile creation date
        followers = tweet.author.followers_count #number of author/user followers (inlink)
        friends   = tweet.author.friends_count   #number of author/user friends (outlink)
        authorid  = tweet.author.id              #author/user ID#
        authorloc = tweet.author.location        #author/user location
        #TECHNOLOGY INFO
        geoenable = tweet.author.geo_enabled     #is author/user account geo enabled?
        source    = tweet.source                 #platform source for tweet
		# Dongho 03/28/16
        csvWriter.writerow([created, text, retwc, hashtag, followers, friends ])
        counter = counter +1
        if (counter == c):
            csvFile.close()
            exit()
'''
        if (d == "sc"):
	        print ("CREATED_AT:", created)
	        print ("TEXT:::::::", text)
	        print ("TWEET_ID:::", tweet_id)
	        print ("COORDS:::::", cords)
	        print ("RETWEET_CT:", retwc)
	        print ("HASHTAG::::", hashtag)
	        print ("URLS:::::::", urls)
	        print ("USER_NAME::", username)
	        print ("USER_SINCE:", usersince)
	        print ("FOLLOWERS::", followers)
	        print ("FRIENDS::::", friends)
	        print ("USER_ID::::", authorid)
	        print ("USER_LOC:::", authorloc)
	        print ("GEO_ENABLE:", geoenable)
	        print ("TECH_SRCE::", source)


        print ("\n")
        counter = counter + 1
        if (counter == c):
            exit()
''' 

# MAIN ROUTINE
def main():

    global api, cmax, locords

    # Geo-coordinates of five metropolitan areas
    # London, NYC (lower, middle, upper), Wash DC, San Francisco, New Brunswick (NJ)
    locords =  {'lo': '0, 51.503, 20km',
                'nyl': '-74, 40.73, 2mi',
                'nym': '-74, 40.74, 2mi',
                'nyu': '-73.96, 40.78, 2mi',
                'dc': '-77.04, 38.91, 2mi',
                'sf': '-122.45, 37.74, 5km',
                'nb': '-74.45, 40.49, 2mi'}
    # Maximum allowed tweet count (note: Twitter sets this to ~180 per 15 minutes)
    cmax = 50
    # OAuth key file
    authfile = './auth.k'

    tw_parser()
    api = tw_oauth(authfile)
    tw_search(api)

if __name__ == "__main__":
    main()
