# Pandas
## Data frame from dictionary, specify row labels
```
import pandas as pd

# Create dictionary out of lists
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
my_dict = { 'country': names, 'drives_right': dr, 'cars_per_cap':cpc }

# Builds the dataframe from a dictionary
cars = pd.DataFrame(my_dict) 

# Define row labels / indexes for each row using an array
row_labels = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']
cars.index = row_labels
```
## Data frame from csv and to csv
```
import pandas as pd
cars = pd.read_csv('cars.csv') # Optional: specify which column should be used as index : index_col = 0 
cars_updated.to_csv('cars_updated.csv') # Write a new csv file with the updated data frame
```
## Getting info about a data frame:
```
df.head() # Prints 1st five rows by default, can provide integer argument for a custom number of 1st rows
df.info() # info about the columns, such as column names and data types
print(df.shape) # Shape of the data frame: (row_nr , col_nr)
df.describe() # summary statistics for each column

df.values # A two-dimensional NumPy array of values.
df.columns # An index of columns: the column names.
df.index # An index for the rows: either row numbers or row names.
```
## Sorting a data frame
```
df.sort_values(["col_name1","col_name2"]) # optional ascending=[True,False] (for each column by which we are sorting)
```
## Indexing
```
# Note: Indexing might be useful, but complicates things and should generally be avoided

# New DF with "name" column values set as the index of the df ( values don't need to be unique! )
dogs_indexed = dogs.set_index("name")

# Reset the index back to normal (numeric indexes)
dogs_indexed.reset_index() # drop=True - drops the column instead of putting it back to a normal column

# Usefulness: dogs[dogs["name"].isin(["Dog1","Dog2"])] => dogs_indexed.loc[["Dog1","Dog2"]]

# Multi-level indexes:
dogs_multi_indexed = dogs.set_index(["name","breed"])
# Usage: dogs_multi_indexed.loc[[("Dog1","Breed1"),("Dog2","Breed2")]]

# Sort by index:
dogs_multi_indexed.sort_index()
dogs_multi_indexed.sort_index(level=["color","breed"], ascending=[True,False])

```
## Fetching data from the pandas data frame
```
cars['country'] # returns a "series" object with index(key) + value pairs, labeled 1d array
cars[['country']] # returns a "data frame" object with the index + single selected column
cars[['country','cars_per_cap']] # returns a "data frame" object with the index + selected columns

cars[1:5] # returns a new data frame - slice of the old data frame or certain rows, works with series as well

cars.loc['RU'] # returns a row as pandas series - row access location by index/label
cars.loc[['RU']] # returns a row as pandas data frame - row access location by index/label
cars.loc[['RU','US']] # returns rows as pandas data frame - row access location by index/label
cars.loc[ ['RU','US'] , ['country','cars_per_cap'] ] # returns pandas frame of selected rows by index/label + selected columns
cars.loc[ : , ['country','cars_per_cap'] ] # all rows, certain columns

cars.iloc[[1]] # returns a row as pandas data frame - row access location by index number
cars.iloc[[1,2,3]] # returns a row as pandas data frame - row access location by index number
cars.iloc[ [1,2,3] , [1,2] ] # returns a row as pandas data frame - row access location by index number + column by column position number
cars.iloc[ : , [1,2] ] # all rows, certain columns
```
## Slicing and subsetting with .loc and .iloc
```
breeds[2:5] # slice a list, get's elements 2,3,4
dogs_srt=dogs.set_index(["breed","color"]).sort_index() # Set a multi-index and sort it
dogs_srt.loc["breed1":"breed4"] # gets all columns for rows where the 1st index values are between x and y
dogs_srt.loc[("breed1","color1"):("breed4","color3")] # gets all columns for rows between the specified tuple values x and y. Tuple in this case = ("breed","color") 
dogs_srt.loc[:,"col1":"col5"] # all rows, certain columns by name
dogs_srt.loc[("breed1","color1"):("breed4","color3"),"col1":"col5"] # certain rows by multi-index tuple values, certain columns by name
dogs=dogs.set_index("date_of_birth:).sort_index() # set index to a date + sort
dogs.loc["2010-01-05":"2010-01-10"] # get rows by date range
dogs.loc["2010":"2011"] # get rows by date range (aproximate/partial, in this case 2010-01-01 <-> 2011-12-31)
dogs.iloc[2:5, 1:4] # certain rows by index id, cetain columns by column number
```
## Summary statistics:
```
# Single value result:
df["column"].mean()
df["column"].median()
df["column"].min()
df["column"].max()

# Custom sumarry statistics:
# Example function
def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)
# Example usage (multiple columns and multiple statistics)
df[['column1','column2']].agg([function1,function2])

# Cumulative (result for each row)
df["column"].cumsum() # cumulative sum
df["column"].cummax() # cumulative max
```
## Duplicate dropping, counting
```
df.drop_duplicates(subset=["col1","col2"]) # distinct on col1, col
df["col"].value_counts() # optional: sort=True ; normalize=True (for proportions)
store_depts["department"].value_counts().sort_values(ascending=False) # another example of sorting by count
store_depts["department"].value_counts().sort_index(ascending=False) # example of sorting by index
print(dept_counts_sorted)
```
## Grouping for aggregate operations, pivot tables
```
df.groupby("col1")["col2"].opperation() # example df.groupby("color")["price"].mean() - average price for each color
df.groupby("col1")["col2"].agg([func1, func2]) # getting multiple aggregates in one call
df.groupby(["col1","col2"])[["col3","col4"]].opperation()
df.groupby("col1")[["col3","col4"]].opperation() # aggregate multiple columns
df.groupby("col1")[["col2","col3"]].opperation() # grouping by several columns
df.groupby(["col1","col2"])[["col3","col4"]].opperation() # group by and aggregate multiple columns

#pivot tables:
dogs.pivot_table( values="weight", index="color", aggfunc=[func1, func2] ) # index=groupby, values=aggregatewhat
#group by 2 columns pivot (gets mean by default), fill NaN with 0:
dogs.pivot_table( values="weight", index="color", columns="breed", fill_value=0, margins=True ) # index=groupby1, columns=groupby2, values=aggregatewhat, fill_value=fillmissingwithsmtng, margins=showsummarystats
pivot_df.mean(axis="index") # "index" is the default value, get's the mean for each column, but "columns" can be used to calculate the mean for each row as well.
#example:
temp_by_country_city_vs_year = temperatures.pivot_table( values="avg_temp_c", index=temperatures["country"]+' '+temperatures["city"], columns="year")
temp_by_country_city_vs_year = temperatures.pivot_table( values="avg_temp_c", index=[temperatures["country"],temperatures["city"]], columns="year")
```
## Logical operations with numpy arrays
```
np.logical_and(number_arr_1 > 13, number_arr_2 < 15) 
np.logical_or(number_arr_1 > 13, number_arr_2 < 15) 
np.logical_not(number_arr_1 > 13, number_arr_2 < 15) 
```
## Basic control flow (if, elif, else)
```
area = 10.0
if(area < 9) :
    print("small")
elif(area < 12) :
    print("medium")
else :
    print("large")
```
## Dealing with missing values:
```
dogs.isna() # check which which columns for which rows has missing values (returs a boolean data frame)
dogs.isna().any() # check which column has any missing values (returns a boolean series)
dogs.isna().sum() # check how many missing values exist in each column (returns a numeric series)
dogs.dropna() # drops rows where any of the columns have a missing value
dogs.fillna(0) # replaces missing values with 0

```
## Filtering (subsetting) pandas data frames
```
cpc = cars['cars_per_cap'] # get a series of numeric values for each country
between = np.logical_and(cpc > 10, cpc < 80) # get a series of boolean values depending on the conditions
medium = cars[between] # select only those countries that have TRUE in the given boolean series

dogs[dogs["height_cm"] > 60]
dogs[dogs["color"] == "tan"]
dogs[(dogs["height_cm"] > 60) & (dogs["color"] == "tan")]

colors = ["brown", "black", "tan"]
condition = dogs["color"].isin(colors)
dogs[condition]
```
## Basic python while and for loops
```
# simple while
x = 1
while x < 4 :
    print(x)
    x = x + 1
    
# array
areas = [1.73, 1.68, 1.71, 1.89] 
for index, value in enumerate(areas) :
    print(str(index) + ': ' + str(value))

# dictionary
world = { "afghanistan":30.55, 
          "albania":2.77,
          "algeria":39.21 }
for key, value in world.items() :
    print(key + " -- " + str(value))
    
# numpy array
np_arr = np.array([2,3,4])
for x in np.nditer(np_arr) :
    ...
    
# pandas data frame
for label, row in brics.iterrows() :
    ...
```
## Loop through pandas data frame and adjust the row
```
# Creates new column and assigns the lenght of another column as value
for lab, row in brics.iterrows() :
    brics.loc[lab, "name_length"] = len(row["country"])

# Same with using apply() - more performant/efficient
brics["name_length"] = brics["country"].apply(len)
```
## Numpy random numbers
```
# Import numpy as np
import numpy as np
import numpy.random as rnd

# Set the seed
rnd.seed(123)

print( rnd.rand() ) # generates a random float between 0 and 1
print( rnd.randint(4, 8) ) # Generates a random integer between 4 and 7 (8 is excluded)
```
## Random number walk exercise
```
# Numpy is imported, seed is set
# Initialize random_walk
random_walk = [0]
for x in range(100) :
    step = random_walk[-1] # Set step: last element in random_walk
    dice = np.random.randint(1,7) # Roll the dice
    # Determine next step
    if dice <= 2:
        step = max(0, step - 1) # using max to make sure it can't go below 0
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)
    random_walk.append(step) # append next_step to random_walk
print(random_walk)
```
## Data frame merging ( JOIN ):
```
# Merging two data frames Default : Inner
new_df = df1.merge( df2, on=["column_name1","column_name2"] ) 
# to add custom suffixes to clumns that exist in both tables : suffixes=('_df1','_df2')
# to change join type (inner,left,right, outer) : how='left'
# if key column names are different : on_left='col1a', on_right='col1b'
# if key column names are different and are index we also need to specify that : left_index=True, right_index=True

# Merging two data frames - ordered! Default : Outer
pd.merge_ordered(df1, df2, [on=..., on_left=..., on_right=..., suffixes=..., how=..., ])
# if we want to fill missing columns for row with value from previous: fill_method='ffill'

# Merging two data frames - ordered, vague, closest!
pd.merge_asof(df1, df2, [on=..., on_left=..., on_right=..., suffixes=..., how=..., direction=... ])
```
## Filtering join and anti-join to data-frames ( WHERE EXISTS / NOT EXISTS )
```
# Filtering join / semi-join - leaving rows in table1 only if they also exist in table2
genres_tracks = genres.merge(top_tracks, on='gid')
genre_exists = genres['gid'].isin(genre_tracks['gid'])
top_genres = genres[genre_exists]

# Another example:
# Merge the non_mus_tck and top_invoices tables on tid
# Gives info about the how many of the non-music tracks were sold
tracks_invoices = non_mus_tcks.merge(top_invoices, on='tid')

# Use .isin() to subset non_mus_tcks to rows with tid in tracks_invoices
# Take only those non-music tracks that had any sales
top_tracks = non_mus_tcks[non_mus_tcks['tid'].isin(tracks_invoices['tid'])]

# Group the top_tracks by gid and count the tid rows
# Count how many tracks had any sales per genre
cnt_by_gid = top_tracks.groupby(['gid'], as_index=False).agg({'tid':'count'})
 
# Merge the genres table to cnt_by_gid on gid and print
# Get more genre info about those genres
genres_and_how_many_tracks_had_sales = cnt_by_gid.merge(genres, on='gid')

# Rename some columns to clarify
genres_and_how_many_tracks_had_sales.rename(columns={
    'tid': 'nr_of_tracks_that_had_sales',
    'name': 'genre_name',
    'gid': 'genre_id'},
    inplace=True)
    
    
# Anti-join - leaving rows in table1 only if they don't exist in table2
genres_tracks = genres.merge(top_tracks, on='gid', how='left', indicator=True) # indicator creates a "_merge" column
gid_list = genres_tracks.loc[ genres_tracks['_merge'] == 'left_only', 'gid' ]
non_top_genres = genres[genres['gid'].isin(gid_list)]
```
## Concatenate / UNION data frames:
```
pd.concat([df1, df2]) 
# ignore_index=True # optional if we want a new fresh index
# keys=['tab1','tab2'] # if we want set labels, makes a multi-index with rows of each table starting from fresh index
# sort=True # sorts the column names alphabetically
# join='inner' # only returns that columns that exist in both tables
df1.append([df2,df3]) # simpler version, only has ignore_index and sort options, always outter join
```
## Verify/validate integrity for merge and concatenate:
```
# merge:
tracks.merge(specs, on='tid', validate='one_to_one') # raises error if the result merge is not one_to_one
# concat:
pd.concat([inv_feb,inv_mar], verify_integrity=True) # raises error if the indexes have overlapping values
```
## Query method : .query() - WHERE clause:
```
stocks.query('stock=="disney" or (stock=="nike" and close < 90)')
```
## Melting : .melt() - change the data frame from wide to long
```
social_fin.melt( id_vars=['financial','company'], value_vars=['2018','2019'], var_name=['year'], value_name='dolars' ) 
# ^ keeps only "financial" and "company" columns original
# ^ changes all other columns to "variable" and "value" column values
# ^ from all other columns only keeps the ones listed under value_vars
# ^ also naming the "variable" and "value" columns
```
## Clean up special characters from text field and cast to numeric:
```
# List of characters to remove
chars_to_remove = ['+',',','$']
# List of column names to clean
cols_to_clean = ['Installs','Price']

# Loop for each column in cols_to_clean
for col in cols_to_clean:
    # Loop for each char in chars_to_remove
    for char in chars_to_remove:
        # Replace the character with an empty string
        apps[col] = apps[col].apply(lambda x: x.replace(char, ''))

# Convert Installs to float data type
apps["Installs"] = apps["Installs"].astype('float')

# Convert Price to float data type
apps["Price"] = apps["Price"].astype('float')
```
## Unique values
```
# Unique values and nr of unique values in a column
unique_categories = apps['Category'].unique()
num_of_unique_categories = apps['Category'].nunique()

# Row count for each unique value in a column 
num_apps_in_category = apps['Category'].value_counts()
```
## Filter out rows where certain columns have null values - Query .notnull()
```
apps_with_size_and_rating_present = apps.query(' Rating.notnull() and Size.notnull() ')
```
## Group up values and filter whole groups
```
# Explanation: Takes each "category" as a separate group/entity, which becomes an array of rows(?) and allows to filter out groups
large_categories = apps_with_size_and_rating_present.groupby('Category').filter(lambda x: len(x) >= 250) # Subset for categories with at least 250 apps
```
## Lambda function + map
```
nums = [48, 6, 9, 21, 1]
square_all = map( lambda num: num**2 , nums )
print(square_all) # <map object...>
print(list(square_all)) # [2304, 36, 81, 441, 1]
```
## Try / Except error and exception handling
```
def sqrt(x):
	if x < 0:
		raise ValueError('x must be non-negative') # Raises an exception
	try:
		return x ** 0.5
	except: # <- can use except TypeError here if we only want to catch those
		print('x must be an int or a float') # Just prints a message
```
## Iterables and iterators
### String example
```
word = 'Data'
word_iterator = iter(word)
next(word_iterator) # 'D'
next(word_iterator) # 'a'
...
word_iterator = iter(word)
print(*word_iterator) # 'D a t a'
```
### Dictionary example
```
some_dict = { 'a': 1, 'b': 2 }
for key, value in some_dict.items():
	print(key, value)
```
### File connection example:
```
file = open('file.txt')
it = iter(file)
print(next(it)) # "This is the first line."
print(next(it)) # "This is the second line."
```
## Enumerate
```
some_list = ['a','b','c']
e = enumerate(some_list)
e_list = list(e)
print(e_list) [(1,'a'),(2,'b'),(3,'c')]

some_list = ['a','b','c']
for index, value in enumerate(some_list, start=10): #starts index from 10
	print(index, value)
```
## Zip
```
some_list_1 = ['a','b','c']
some_list_2 = ['ss','ig','ut']
z = zip(some_list_1, some_list_2)
z_list = list(z)
print(z_list) # [('a','ss'),('b','ig'),('c','ut')]
print(*z) # ('a','ss'),('b','ig'),('c','ut')

some_list_1 = ['a','b','c']
some_list_2 = ['ss','ig','ut']
for z1, z2 in zip(some_list_1, some_list_2):
	print(z1, z2)
	
z = zip(some_list_1, some_list_2)
list1, list2 = zip(*z)
print(list1 == some_list_1) # true
print(list2 == some_list_2) # true
```
## Loading large files into memory - loading data in chunks 
```
result=[]
for chunk in pd.read_csv('data.csv', chunksize=1000):
	result.append(sum(chunk['x']))
total = sum(result)
# or
total=0
for chunk in pd.read_csv('data.csv', chunksize=1000):
	total += sum(chunk['x'])
```
```
# Open a connection to the file
with open('world_dev_ind.csv') as file:

    # Skip the column names
    file.readline()

    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Process only the first 1000 rows
    for j in range(1000):

        # Split the current line into a list: line
        line = file.readline().split(',')

        # Get the value for the first column: first_col
        first_col = line[0]

        # If the column value is in the dict, increment its value
        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1

        # Else, add to the dict and set value to 1
        else:
            counts_dict[first_col] = 1

# Print the resulting dictionary
print(counts_dict)
```

## List comprehensions - shorter/single-line for loops
```
nums = [3,5,7,8]
new_nums = [num + 1 for num in nums]

pairs = [ (num1,num2) for num1 in range(0,2) for num2 in ranger(6,8) ]
print(pairs) # [ (0,6), (0,7), (1,6), (1,7) ]

matrix = [[col for col in range(5)] for row in range(5)]
```
### Conditionals in comprehensions
```
nums = [num ** 2 for num in range(10) if num % 2 == 0]
nums = [num ** 2 if num % 2 == 0 else 0 for num in range(10) ]
```
### Dictionary comprehensions
```
pos_neg = { num: -num for num in range(9) }
```
## Generators - same as comprehension, but does not produce a list, just stores the defintion as a generator object.
```
# Generators can be iterated over to generate the values on demand. Or converted to a list to materlialize the definition.
# Basically avoids storing the whole result in list, rather generates elements on the fly by demand. Good for generating large data sets.
gen = (num for num in range(6))
for num in gen: 
	print(num)
print(list(gen))
print(next(gen)) # 0
print(next(gen)) # 1
```
### Generator functions and keyword "yield"
```
def num_sequence(n):
	i = 0
	while i < n:
		yeald i
		i += 1
result = num_sequence(5)
for item in result:
	print(item)
```


# Importing data

## Importing Text files

### Reading a text file
```
filename = 'xxx.txt'

# Open file for reading, close manualy
file = open(filename, mode='r') # r=read, w=write
print(file.closed) # check if the file is closed or not
text = file.read()
file.close()
print(text)

## open using a context manager 

with open(filename, mode='r') as file: # context manager , binding a variable inside context manager construct
	text = file.read()
	print( text )
	
# read line by line
with open('moby_dick.txt') as file:
    print(file.readline())
    print(file.readline())
    print(file.readline())
```

### Importing/reading a flat file
Flat files : Text files containing table data, contains header, records (rows of fields or attributes), columns(feature or attribute)

Importing with NumPy:
```
# Importing numeric flat file using NumPy : loadtxt() and genfromtxt()
import numpy as np

filename = '...'
data = np.loadtxt( filename, delimiter=',' ) # skiprows=1, usecols=[0,2], dtype=str
print(data)

data = np.genfromtxt('titanic.csv', delimiter=',', names=True, dtype=None) # names=True means we have headers, dtype=None allows implicit data type assigning
```

Importing flat files / csv with Pandas:
```
import pandas as pd
data = pd.read_csv('my-csv-file.csv') # nrows = 5, header=None, sep='\t', comment='#', na_values='Nothing'
data.head()
```

### Importing "other" files:

Pickled files:
```
import pickle
with open('pickled_fruit.pkl', 'rb') as file: # rb = read only, binary
	data = pickle.load(file)
print(data)
```

Importing Excel spreadsheets:
```
import pandas as pd
file = 'xxx.xlsx'
data = pd.ExcelFile(file)
print(data.sheet_names)
df1 = data.parse('some_sheet_name') # sheet name as a text
df2 = data.parse(0) # sheet index as a float
# some other arguments for data.parse(...):
# skiprows=[0], names=['Country','AAM due to War (2002)'], usecols=[0]
```

Importing SAS and stata files:
```
Import pandas as pd
from sas7bdat import SAS7BDAT
with SAS7BDAT('xxx.sas7bdat') as file:
	df_sas = file.to_data_frame()
	
data = pd.read_stata('xxx.dta') # stata files
```

Importing HDF5 (large quantities of numerical data, terabytes, can scan to exabytes)
```
import h5py
filename = 'xxx.hdf5'
data = h5py.File(filename, 'r') # read
```

Importing MATLAB files:
```
import scipy.io
filename = 'xxx.mat'
mat = scipy.io.loadmat(filename)
```


### Querying relation databases
```
from sqlalchemy import create_engine
engine = create_engine('sqlite:///Northwind.sqlite')
print(engine.table_names()) # check tables

con = engine.connect()
rs = con.execute('SELECT * FROM...')
df = pd.DataFrame(rs.fetchall())
df.columns = rs.keys()
con.close()

or 

with engine.connect() as con:
	rs = con.execute('SELECT * FROM...')
	df = pd.DataFrame(rs.fetchall())
	df.columns = rs.keys()
	
or

df = pd.read_sql_query('SELECT * FROM ...', engine)
```


# Importing data from the web

## HTTP requests

### urllib package , urlopen()
```
from urllib.request import urlretrieve

#csv
url = 'http:.../xxx.csv'
urlretrieve(url, 'xxx.csv') # save the file locally
df = pd.read_csv('xxx.csv', sep=';')

#xls
url = 'http:.../xxx.xls'
xls = pd.read_excel(url, sheet_name = None)
```

### urllib package , GET request using urlopen() and Request()
```
from urllib.request import urlretrieve, Request
url = 'https...'
request = Request(url)
response = urlopen(request)
html = response.read()
response.close()
```

### Get request using requests package
```
import requests
url = 'https...'
r = requests.get(url)
text = r.text
```

### Using Beautiful soup
```
from bs4 import BeautifulSoup
import requests
url = '...'
r = requests.get(url)
html_doc = r.text
soup = BeautifulSoup(html_doc)

pretty_soup = soup.prettify()
soup_title = soup.title
soup_text = soup.get_text()
a_tags = soup.find_all('a') # fetch all hyperlinks

for link in a_tags:
 print(link.get('href')) # print all hyperlink urls
```

## APIs and JSONS

### Loading JSONs in python
```
import json
with open('xxx.json', 'r') as json_file:
	json_data = json.load(json_file)
	
for key, value in json_data.items():
	print( key + ':', value)
```

### Connecting to an API
```
import requests
url = '...?t=hackers' # query string ?t=hackers
r = requests.get(url)
json_data = r.json()
for key, value in json_data.items():
	print( key + ':', value)
```


# Cleaning data

### Check column info
```
print(df.dtypes) # check data types of each column of the data frame
df.info() # check data frame columns, types and missing values per column
print(df['some_column'].describe()) # describe a specific column
```

### Changing data type of a column
```
df['revenue'] = df['revenue'].str.strip('$') # remove $ signs from the start or end of the string
df['revenue'] = df['revenue'].astype('int') # cast string to integer
assert df['revenue'].dtype == 'int' # check if the data type is now correct, throws error if given condition is not met

df['marriage_status'] = df['marriage_status'].astype('category') # convert integer (in this case) into a "category"

df['some_date_column'] = pd.to_datetime( df['some_date_column'] ) # convert object(string) to datetime
```

### Check if date is in the future
```
import datetime as dt 
today_date = dt.date.today()
df[ df['some_date_column'] > today_date ] # selects only rows that are later than current date
``` 

### Drop / fix data that is out of bounds in some way
```
movies = movies[ movies['rating'] <= 5 ] # only keep movies where the rating is equal or lower to the logical maximum of 5
or
movies.drop( movies[ movies['rating'] <= 5 ].index, inplace = True ) # drop the rows that don't meed a certain condition

assert movies['rating'].max() <= 5 # check if there are no more out of bounds values

movies.loc[ movies['rating'] > 5, 'rating' ] = 5 # assign the value 5 to any rows where the value is above 5
```

### Identifying and dealing with duplicates
```
print( df.duplicated() ) # gives true/false for each row whether an exact same row exists elsewhere in the data frame
print( df[ df.duplicated() ] ) # shows the rows that are duplicated

# arguments for the duplicated() and drop_duplicates() methods : 
# subset=['a','b'] - which columns to check for duplication
# keep='first'/'last'/False - instructs which duplicates to keep (False = all)
# inplace=True - for drop_duplicates() - drops in place
# Example wiht sorted..
df_duplicates_sorted = df[ df.duplicated( subset=['first_name','last_name'], keep=False ) ].sort_values(by = 'first_name')

df.drop_duplicates(inplace = True) # drop complete duplicates in place

# Assigning a value for duplicate rows/columns using .groupby() and .agg(), for example keeping an average of two weights for a person
column_names = ['first_name', 'last_name']
summaries = { 'height': 'max', 'weight': 'mean' }
df = df.groupby( by = column_names ).agg(summaries).reset_index()
```

### Identifying and dealing with rogue category members (using anti joins and inner joins)
```
# Finding inconsistent categories example
inconsistent_categories = set( df['blood_type'] ).difference( category_df['blood_type'] ) # fetch the categories that don't exist in the correct category df
print(inconsistent_categories) # show which categories exist in df which dont exist in category_df
inconsistent_rows = df['blood_type'].isin(inconsistent_categories) # fetch the rows with inconsistent categories, returns boolean values
print(df[inconsistent_rows]) # shows rows that have inconsistent categories
consistent_df = df[~inconsistent_rows] # create new df with only the consistent rows
```

### Identifying and fixing data consistency
```
marriage_status = df['marriage_status'] # returns series of values
marriage_status.value(counts) # returns count of each value in series
or
df.groupby('marriage_status').count() # returns count of each value in a column of a data frame

df['marriage_status'] = df['marriage_status'].str.lower() # assign lower values of string to deal with capitalization inconsistency
df['marriage_status'] = df['marriage_status'].str.strip() # remove leading and trailing spaces from strings
```

### Collapsing data into categories
```
group_names = ['0-200k','200-500k','500k+']

# using qcut()
df['income_group'] = pd.qcut( df['income'], q=3, labels=group_names ) # cut into 3 parts using qcut() <- BAD

# using cut()
ranges = [0, 200000, 500000, np.inf]
df['income_group'] = pd.cut( df['income'], bins=ranges, labels=group_names ) # cut into given ranges using cut()
```

### Cutting number of categories
```
mapping = { 'a':'vowel','b':'consonant','c':'consonant' } # create the mapping
df['letter'] = df['letter'].replace(mapping) # assign mapped values
print( df['letter'].unique() ) # check unique values
```

### Cleaning text data
```
# Phone number cleaning example
df["phone_num"] = df["phone_num"].str.replace("+", "00") # Replace + with 00
digits = df["phone_num"].str.len() # grab number of digits for each string value
df.loc[ digits < 10, "phone_num" ] = np.nan # assign NaN where the value is less than 10 digits
digits = df["phone_num"].str.len() # grab number of digits for each string value again
assert digits.min() >= 10 # use assert that there are no more phone numbers with less than 10 digits
assert df["phone_num"].str.contains("+|-").any() == False # check if any phone numbers still have +'s or -'s

# Regexp example
df["phone_num"] = df["phone_num"].str.replace(r'\D+', '') # regexp to replace non digits with empty string
```

### Converting farenheit to celsius
```
df_fah = df.loc[ df["temp"] > 40, "temp" ]
df_cels = (df_fah - 32) * (5/9)
df.loc[ df["temp"] > 40, "temp" ] = df_cels
```

### Unifying date formats
```
df['bday'] = pd.to_datetime( df['bday'], infer_datetime_format=True, errors = 'coerce' ) # tries to infer format and assigns NaT for rows where it could not
df['bday'] = df['bday'].dt.strftime("%d-%m-%Y") # change date format
```

### Coverting euros to dollars
```
acct_eu = banking['acct_cur'] == 'euro' # Find values of acct_cur that are equal to 'euro'
banking.loc[acct_eu, 'acct_amount'] = banking.loc[acct_eu, 'acct_amount'] * 1.1 # Convert acct_amount where it is in euro to dollars
banking.loc[acct_eu, 'acct_cur'] = 'dollar' # Unify acct_cur column by changing 'euro' values to 'dollar'
assert banking['acct_cur'].unique() == 'dollar' # Assert that only dollar currency remains
```

### Crossfield validation 
```
# Summing up various flight class ticket numbers to see if it matches total passangers
sum_classes = flights[ [ '1st class', '2nd class', '3rd class' ] ].sum(axis = 1)
passenger_equ = sum_classes == flights['total_passengers']
incosistent_pass = flights[~passenger_equ]
consistent_pass = flights[passenger_equ]

# Validating age using birth date
import datetime as dt
users['bday'] = pd.to_datetime( users['bday'] ) # convert to dtime
today = db.date.today() # get current date
age_manual = today.year - users['bday'].dt.year # calculate age
age_equ = age_manual == users['age'] # check where they match
inconsistent_age = users[~age_equ] # filter inconsistent rows
consistent_age = users[age_equ] # filter consistent rows
print("Number of inconsistent ages: ", inconsistent_age.shape[0])
```

# Data completeness - dealing with missing data

### Looking for missing values
```
df.isna() # returns rows with True/False for each column depending on whether the value is missing or not
df.isna().sum() # returns a number for each column of how many rows have missing data in that column

# Get a better picture using describe
missing = df[df['some_column'].isna()]
complete = df[~df['some_column'].isna()]
missing.describe()
complete.describe()
```

### Visualizing missing values with missingno
```
import missingno as msno
import matplotlib.pyplot as plt
msno.matrix(df)
plt.show()

#or with sorted.. ex missing inv_amount seems to be correlated with age
banking_sorted = banking.sort_values(by = 'age')
msno.matrix(banking_sorted)
plt.show()
```
Missingness types:
Missing Completely at Random (MCAR) - missing data independent of other attributes/values
Missing at Random (MAR) - systemic relationship between missing and other observed values (for example missing CO2 only for low temperatures)
Missing Not at Random (MNAR) - systemic relationship between missing data and unobserved values (for example missing temp data for high temperatures)

### Dropping / replacing missing values
```
co2_mean = df['co2'].mean()
df_dropped = df.dropna( subset = ['co2'] ) # drops rows where the value for specified column is missing
df_replaced = df.fillna( {'co2': co2_mean} ) # replaces rows with mean
```

# Record linkage - calculating similarity between strings to link records
minimum edit distance - how close the two strings are
example using Levenshtein method and fuzzywuzzy package
```
from fuzzywuzzy import fuzz
fuzz.WRation('Reeding','Reading') # similarity ratio = 86, 0 means not similar at all, 100 means exact match
# also possible to compare string with array of strings and get a sorted array of tupples including similarty rank and ratio
# Example with linking free text state values (Cali,California,Calefornia, New York, New York City..) with pre-defined categories (California, New York)
from fuzzywuzzy import process
for state in categories['state']: # correct categories
	matches = process.extract( state, survey['state'], limit = survey.shape[0] ) # find potential matches for that category in the df column
	for potential_match in matches:
		if potential_match[1] >= 80: # if match is at least 80%
			survey.loc[ survey['state'] == potential_match[0], 'state' ] = state # replace the state with the pre-defined category
```

Record pair linking with the recordlinkage package
Example: 2 census data sets with person data like name,surname,dateofbirth,suburd,state,address,etc...
Using "blocking" to reduce the number of pairs generated and improve performance - for example only match records from the same state
```
# Import recordlinkage
import recordlinkage

# Create indexing object
indexer = recordlinkage.Index()

# Generate pairs blocked on state
indexer.block('state')
pairs = indexer.index(census_A, census_B)

# Create a Compare object
compare_cl = recordlinkage.Compare()

# Find exact matches for pairs of date_of_birth and state
compare_cl.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
compare_cl.exact('state', 'state', label='state')

# Find similar matches for pairs of surname and address_1 using string similarity
compare_cl.string('surname', 'surname', threshold=0.85, label='surname')
compare_cl.string('address_1', 'address_1', threshold=0.85, label='address_1')

# Find matches
potential_matches = compare_cl.compute(pairs, census_A, census_B)

# Only take matches where 3 or more columns match (exact or partial)
matches = potential_matches[potential_matches.sum(axis = 1) => 3]

# Get indices from census_B only
duplicate_rows = matches.index.get_level_values(1)

# Isolate rows in census_B that are not duplicates of census_A
census_B_new = census_B[~census_B.index.isin(duplicate_rows)]

# Link the two data frames
full_census = census_A.append(census_B_new)
```




# Working with Dates and Times

### Dates
- The date() class takes a year, month, and day as arguments
- A date object has accessors like .year , and also methods like .weekday()
- date objects can be compared like numbers, using min() , max() , and sort()
- You can subtract one date from another to get a timedelta
- To turn date objects into strings, use the .isoformat() or .strftime() methods
```
from datetime import date

# Extract month from dtime
print( some_dtime.month ) 

# Order an array of dates
dates_ordered = sorted(dates_scrambled)

 # Subtract the two dates and print the number of days
start = date(2007, 5, 9)
end = date(2007, 12, 13)
print((end - start).days)

# Convert to ISO and US formats
iso = "Our earliest hurricane date: " + first_date.isoformat()
us = "Our earliest hurricane date: " + first_date.strftime("%m/%d/%Y")
```

### Datetimes
- The datetime() class takes all the arguments of date() , plus an hour, minute, second, and microsecond
- All of the additional arguments are optional; otherwise, they're set to zero by default
- You can replace any value in a datetime with the .replace() method
- Convert a timedelta into an integer with its .total_seconds() method
- Turn strings into dates with .strptime() and dates into strings with .strftime()
```
from datetime import datetime

# Create a datetime object
dt = datetime(2017, 10, 1, 15, 26, 26)

# Replace the year with 1917
dt_old = dt.replace(year=1917)

# Print the results in ISO 8601 format, which is basically "%Y-%m-%dT%H:%M:%S"
print(dt.isoformat())

# String into datetime using strptime and format:
s = '2017-02-03 00:00:01' # Starting string, in YYYY-MM-DD HH:MM:SS format
fmt = "%Y-%m-%d %H:%M:%S" # Write a format string to parse s
d = datetime.strptime(s, fmt) # Create a datetime object d

# Get datetime from timestamp:
ts = 1514665153
dtime = datetime.fromtimestamp(ts)
```
### Durations:
```
# Extract duration in seconds (timedelta)
trip_duration = trip["end"] - trip["start"]
trip_length_seconds = trip_duration.total_seconds()

# Sum total duration using timedelta array
total_elapsed_time = sum(onebike_durations)

# Number of separate timedelta objects in the array
number_of_trips = len(onebike_durations)
  
# Average duration time timedelta array
print(total_elapsed_time / number_of_trips)

# Calculate shortest and longest trips (durations)
shortest_trip = min(onebike_durations)
longest_trip = max(onebike_durations)
```

### Timezone aware datetimes
- A datetime is "timezone aware" when it has its tzinfo set. Otherwise it is "timezone naive"
- Seing a timezone tells a datetime how to align itself to UTC, the universal time standard
- Use the .replace() method to change the timezone of a datetime , leaving the date and time the same
- Use the .astimezone() method to shi the date and time to match the new timezone 
- dateutil.tz provides a comprehensive, updated timezone database
```
from datetime import datetime, timedelta, timezone
from dateutil import tz

dt = datetime(2017, 10, 1, 15, 26, 26, tzinfo=timezone.utc) # 2017-10-01T15:26:26+00:00

pst = timezone(timedelta(hours=-8))
dt = datetime(2017, 10, 1, 15, 26, 26, tzinfo=pst) # 2017-10-01T15:26:26-08:00

# Show dtime in UTC:
dt_as_utc = dt.astimezone(timezone.utc)
print('Original:', dt, '| UTC:', dt_as_utc.isoformat()) # Original: 2017-10-01 15:23:25-04:00 | UTC: 2017-10-01T19:23:25+00:00


# Create a timezone object and replace timezone of a datetime object
et = tz.gettz('America/New_York')
some_dtime = some_dtime.replace(tzinfo=et)


# Local vs nonlocal (UK) example:
# Create the timezone object
uk = tz.gettz('Europe/London')

# Pull out the start of the first trip
local = some_dtime

# What time was it in the UK?
notlocal = local.astimezone(uk)

# Print them out and see the difference
print(local.isoformat())
print(notlocal.isoformat())

# Add hours to a date
start = datetime(2017, 3, 12, tzinfo = tz.gettz('America/New_York'))
end = start + timedelta(hours=6)
```

### Reading date and time data in Pandas
- When reading a csv, set the parse_dates argument to be the list of columns which should be parsed as datetimes
- If seing parse_dates doesn't work, use the pd.to_datetime() function
- Grouping rows with .groupby() lets you calculate aggregates per group. For example,.first() , .min() or .mean().resample() groups rows on the basis of a datetime column, by year, month, day, and soon
- Use .tz_localize() to set a timezone, keeping the date and time the same
- Use .tz_convert() to change the date and time to match a new timezone
```
import pandas as pd

# Reading in columns as dates not as strings from CSV
rides = pd.read_csv( 'xxx.csv', parse_dates = ['start_date','end_date'] )
or
rides['start_date'] = pd.to_datetime( rides['start_date'], format = '%Y-%m-%d %H:%M:%S' )

# Creating a duration column
rides['duration'] = rides['end_date'] - rides['start_date'] # results in a timedelta column
rides['duration'].dt.total_seconds().head(5) # show as seconds
```
### Summarizing datetime data in Pandas data frames
```
rides['duration'].mean()
rides['duration'].sum()
rides['duration'].sum() / timedelta(days=91) # out of 91 days, how long (0-100%) of that time is the total duration?

rides['member_type'].value_counts() # how many times each unqiue member type used the bike
rides['member_type'].value_counts() / len(rides) # What % of the time was used by each member type? (0-100%)

rides['duration_in_seconds'] = rides['duration'].dt.total_seconds()
rides.groupby('member_type')['duration_in_seconds'] # how much time did each member group use the bike

rides.resample('M', on = 'start_date')['duration_in_seconds'].mean() # how long was the average ride for each month?
```

### Extra methods for working with datetime
```
# Localizing a timezone-unaware data frame datetime column to some timezone
rides['start_date'] = rides['start_date'].dt.tz_localize('America/New_York', ambiguous = 'NaT') # sets unclear results as NotaTime

# Converting an already localized timestamp
rides['start_date'] = rides['start_date'].dt.tz_convert('Europe/London')

# Return some part of the datetime
rides['start_date'].dt.year # returns year
rides['start_date'].dt.day_name() # returns day of the week in text (Monday)

# Shifting rows around
rides['end_date'].shift(1) # shifts the end_date column one row forward (good for aligning start date with end date..)
```



