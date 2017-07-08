
# coding: utf-8

# In[1]:

# import packages: 

import xml.etree.cElementTree as ET
import pprint
from collections import defaultdict  # available in Python 2.5 and newer
import re
import codecs
import json
from pymongo import MongoClient
import os
import math


# In[2]:

OSM_FILE = "pittsburgh_pennsylvania.osm"
SAMPLE_FILE = "sample_sample_pittsburgh_pennsylvania.osm"
DB_NAME = "pittsburgh_pennsylvania"
JSON_FILE = "pittsburgh_pennsylvania.osm.json"
COLLECTION_NAME = "pittsburgh_pennsylvania_data"


# In[3]:

# get all street types 
street_types = defaultdict(int)
street_data = defaultdict(int)

# lists of expected street types, street mapping, street types, and street suffix:
# consturcted after iterative autiding: 

expected_street_types = ["Street", "St.", "St", "ST",
                         "Avenue", "Av" , "Ave" , "Ave.", "Av.", 
                         "Boulevard", "Blvd", "Boulevard",
                         "Drive", "Dr", "DR", "Dr.",
                         "Court", "Ct.", "Ct",
                         "Place", "Plaza",
                         "Square", "Sq", 
                         "Lane", "Ln",
                         "Way", "Circle", "Alley", "Harbor", "Pike",
                         "Road", "Rd", "Rd.", "rd",
                         "Trail", "Tr", 
                         "Parkway", "Commons", "Hill", 
                         "Highway", "Hwy", "Expressway", 
                         "Terrace", "Ter", "Brdg", "Bridge", "Route"]

street_mapping = { "St": "Street",
            "St.": "Street",
            "ST": "Street",
            "Ave": "Avenue",
            "Av": "Avenue",
            "Av.": "Avenue",
            "Ave.": "Avenue",
            "Rd.": "Road",
            "Rd": "Road",
            "rd": "Road",
            "Dr" : "Drive",
            "DR" : "Drive",
            "Dr." : "Drive",
            "Sq" : "Square", 
            "Hwy" : "Highway", 
            "Ct." : "Court",
            "Ct" : "Court", 
            "Blvd": "Boulevard",
            "Tr" : "Trail",
            "Ln" : "Lane",
            "Ter": "Terrace",
            "Brdg": "Bridge"}

correct_street_types = ["Street",
                         "Avenue", 
                         "Boulevard",
                         "Drive", 
                         "Court", 
                         "Place", 
                         "Square", 
                         "Lane", "Way", "Circle", "Alley", "Harbor", "Pike",
                         "Road",  
                         "Trail", "Parkway", "Commons", "Hill", 
                         "Highway", "Plaza", "Expressway", "Terrace", "Route"]

street_suffix = ["Extension", "North", "South", "East", "West"]


# In[4]:


# this method audits street type by street name by looking for last word abbrv. and validates it 
# against the expected values: 

def audit_street_type(street_name):
    suffix = ""
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in street_types:
            street_types[street_type] += 1 
            street_data['street_types'] += 1
        if street_type in street_suffix: 
            suffix = street_type
            # call for method to validate street types with suffix: 
            street_name , street_type = audit_street_with_suffix(street_name, street_type)
        return street_name, street_type, suffix
            
def is_street_tag(element): 
    return element.attrib['k'] == 'addr:street'


# In[5]:

# method to validate street types with suffix: 
def audit_street_with_suffix (street_name, street_type): 
        street_name = street_name.replace(street_type, "").rstrip()
        street_name, street_type, suffix = audit_street_type(street_name)
        street_data['expected_streets_with_suffix'] += 1
        return street_name, street_type


# In[6]:

street_type_re = re.compile(r'\S+\.?$', re.IGNORECASE)
street_names = defaultdict(int)
street_types_unexpected = defaultdict(int)

# method to audit street names and collects info about street data: 
def audit_street_names (filename):
    for _, element in ET.iterparse(filename):
        if element.tag == 'tag':
            if is_street_tag(element): 
                street_data['total_number_of_streets'] += 1
                street_name = element.attrib['v']
                street_name, street_type, suffix = audit_street_type(street_name)
                if street_type in expected_street_types:
                    street_data['expected_streets'] += 1
                    update_street_name(street_mapping, street_name, suffix)
                else: 
                    street_data['unexpected_streets'] += 1
                    street_types_unexpected[street_type] += 1
                street_names[street_name] += 1
    print street_types_unexpected
    print street_data


# In[7]:

postal_codes = defaultdict(int)

def audit_postal_codes (filename):
    for _, element in ET.iterparse(filename):
        if element.tag == 'tag':
            if element.attrib['k'] == 'addr:postcode': 
                postcode = element.attrib['v']
                postal_codes[postcode] += 1
    print postal_codes


# In[8]:

# method to update street names that need correction based on supplied mappaing: 
def update_street_name (mapping, street_name, suffix):
    name = street_name
    street_type = street_type_re.search(street_name).group() 
    if  street_type not in correct_street_types:
        street_data['incorrect_streets'] += 1
        name = street_name.replace(street_type, mapping[street_type])
        street_data['corrected_streets'] += 1
    if suffix in street_suffix:
        name = name + " - " + suffix
        street_data['corrected_streets_suffix'] += 1
    return name


# In[9]:

zip_code_re = re.compile(r'\d+', re.IGNORECASE) 

def update_postal_code (postcode):
    m = zip_code_re.search(postcode)
    if m:
        match = m.group(0)
        return match


# In[10]:

if __name__ == '__main__':
    audit_street_names(OSM_FILE)
    audit_postal_codes(OSM_FILE)


# In[11]:

# method to audit and get address tags: 
def audit_address_tags(filename): 
    address_tags = set()
    for _, element in ET.iterparse(filename):
        if element.tag == 'tag': 
            if element.attrib['k'].startswith('addr:'):
                address_tags.add(element.attrib['k'])
    print address_tags
    return address_tags


# In[13]:

if __name__ == '__main__':
    audit_address_tags(OSM_FILE)


# In[14]:

# list of address keys to be chosen for JSON data extract, cosntructed afrer auditing address tags: 
address_keys = ['housenumber', 'postcode', 'housename', 'street', 'city', 'county', 'state', 'country']
address_re = re.compile(r'addr:(.*)', re.IGNORECASE)
address = {}

# method to get address data and add it to dictionary object: 
def get_address_data(item): 
    key = item.attrib['k']
    m = address_re.search(key)
    if m:
        address_key = m.group(1)
        if address_key in address_keys:
            value = item.attrib['v']
            if address_key == 'street':
                street_name, street_type, suffix = audit_street_type(value)
                if street_type in expected_street_types:
                    value = update_street_name(street_mapping,street_name, suffix) #correct street name
            if address_key == 'postcode':
                value = update_postal_code(value)
            address[address_key] = value    
    return address


# In[15]:

CREATED = [ "version", "changeset", "timestamp", "user", "uid"]
data_tags = ["name", "shop", "amenity", "cuisine"]
# method to format data element to JSON node: 
# the data will be in this format: 
"""
{
  "created": {
    "changeset": "22220931", 
    "user": "emacsen_dwg", 
    "version": "5", 
    "uid": "1782960", 
    "timestamp": "2014-05-08T23:20:17Z"
  }, 
  "pos": [
    "-79.9187569", 
    "40.5160917"
  ], 
  "visible": "true", 
  "address": {
    "city": "Penn Hills", 
    "state": "PA", 
    "street": "Kittanning Pike", 
    "housenumber": "415", 
    "postcode": "15235"
  }, 

"""
    
def shape_element(element):
    node = {}
    position = []
    created = {}
    if element.tag == "node" or element.tag == "way" :
        node = get_first_level_element_data(element)
        return node

# method to proccess element first level data into node: 
def get_first_level_element_data(element):
    node = {}
    position = []
    created = {}
    node['type'] = element.tag
    node['visible'] = "true"
    for attrib in element.attrib:
        if attrib == 'lat' or attrib == 'lon': 
            position.append(element.attrib[attrib])
            node['pos'] = position
        elif attrib in CREATED:
            created[attrib] = element.attrib[attrib]
            node['created'] = created
        else: 
            node[attrib] = element.attrib[attrib]
        get_sub_element_data(element, node)
    return node

    
# method to process sub element data into node: 
def get_sub_element_data (element, node): 
    node_refs = []
    address = {}
    for item in element.iter():
        if item.tag == 'nd':
            node['node_refs'] = node_refs.append(item.attrib['ref']) 
        elif item.tag == 'tag': 
            node['address'] = get_address_data(item)     
            key = item.attrib['k']
            if key in data_tags: 
               node[key]  =  item.attrib['v']


# In[16]:

# method to iterate though .OSM file and write its data to JSON file: 
def write_json(file_in, pretty = False):
    file_out = "{0}.json".format(file_in)
    data = []
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data


# In[172]:

if __name__ == '__main__':
    data = write_json(OSM_FILE, True)


# In[1]:

# method to get database by its name: 
def get_db(db_name):
    # For local use
    client = MongoClient('localhost:27017')
    db = client[db_name]
    return db


# In[8]:

# info about the data: 
if __name__ == "__main__":
    db = get_db(DB_NAME)
    print "Database Name: " + db.name
    
    print "Sample Data Point: "
    pprint.pprint (db.pittsburgh_pennsylvania_data.find_one())
    
    print "Data Statisisc: "
    pprint.pprint(get_overview_statistics(OSM_FILE, JSON_FILE, db))


# In[6]:

def get_overview_statistics(osm_filename, json_filename, db):
    statistics = {}
    statistics['OSM file size - MB'] = os.path.getsize(osm_filename) * (1e-6)
    statistics['JSON file size - MB'] = os.path.getsize(json_filename) * (1e-6)
    statistics['Total documents count'] = db.pittsburgh_pennsylvania_data.find().count()
    statistics['Total nodes count'] = db.pittsburgh_pennsylvania_data.find({"type": "node"}).count()
    statistics['Total ways count'] = db.pittsburgh_pennsylvania_data.find({"type": "way"}).count()
    statistics['Total unique users'] = len(db.pittsburgh_pennsylvania_data.distinct("created.uid"))
    
    return statistics


# In[76]:

def query_db (db):    
    print "Top Contributing User: "
    top_users = db.pittsburgh_pennsylvania_data.aggregate([{"$group":{"_id":"$created.uid", "count":{"$sum":1}}}, {"$sort":{"count":-1}}, {"$limit":1}])
    for user in top_users:
        print user

    print "Top Type of Amenities: "
    top_amenities = db.pittsburgh_pennsylvania_data.aggregate([{"$match":{"amenity":{"$exists":1}}}, {"$group":{"_id":"$amenity","count":{"$sum":1}}}, {"$sort":{"count":-1}}, {"$limit":1}])
    for amenity in top_amenities:
        pprint.pprint(amenity)
        
    print "Top Type of Shops: "     
    top_shops = db.pittsburgh_pennsylvania_data.aggregate([{"$match":{"shop":{"$exists":1}}}, {"$group":{"_id":"$shop","count":{"$sum":1}}}, {"$sort":{"count":-1}}, {"$limit":1}])
    for shop in top_shops:
        pprint.pprint(shop)
     
    print "Top Type of cuisines: "
    top_cuisines = db.pittsburgh_pennsylvania_data.aggregate([{"$match":{"cuisine":{"$exists":1}}}, {"$group":{"_id":"$cuisine","count":{"$sum":1}}}, {"$sort":{"count":-1}}, {"$limit":1}])
    for cuisine in top_cuisines:
        pprint.pprint(cuisine)
    
    print "Most Popular Shops: "
    shop_results = db.pittsburgh_pennsylvania_data.aggregate([{"$match":{"shop":{"$exists":1}, "name":{"$exists":1}}}, {"$group":{"_id":"$name","count":{"$sum":1}}}, {"$sort":{"count":-1}},{"$limit":3}])
    for result in shop_results:
        pprint.pprint(result)
            


# In[77]:

# get additional info about db: 
if __name__ == "__main__":
    db = get_db(DB_NAME)
    query_db(db)


# In[80]:

print "Education Stats: "
edu_results = db.pittsburgh_pennsylvania_data.aggregate([{"$match":{"amenity":{"$exists":1}, "amenity": {'$in': ["university","school", "college", "grade_school", "childcare", "education_centre"]}}}, {"$group":{"_id":"$amenity","count":{"$sum":1}}}, {"$sort":{"count":-1}}, {"$limit":10}])
for result in edu_results:
    pprint.pprint(result)
    
    
school_sample = db.pittsburgh_pennsylvania_data.find_one({"amenity": "school"})
print "Sample School: "
pprint.pprint (school_sample)
nearby_houses = db.pittsburgh_pennsylvania_data.find ({"address.street": school_sample["address"]["street"]}, {"type": "node"}).count()
print "Nearby houses count: "
print nearby_houses

