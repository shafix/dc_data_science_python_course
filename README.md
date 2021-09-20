# dc_data_science_python_course
Code used when going through the DataCamp "Data Scientist with Python" course





# Matplotlib
## line plot
```
import matplotlib.pyplot as plt
plt.plot(x,y)

or

df.plot(x="date",y="weight",kind="line") # rot=45 - rotates x axis label by 45 degrees
```
## scatter plot
```
import matplotlib.pyplot as plt
plt.scatter(x,y) # Optional: s = bubble_size_arr ; c = bubble_color_dict ; alpha = opacity_float
plt.xscale('log') # changes the plot to logarithmic scale

or 

df.plot(kind="scatter", x="height", y="weight")
```
## histogram
```
import matplotlib.pyplot as plt
plt.hist(data,bins)

or 

df["height"].hist(bins=5)

# Layering 2 histograms
dogs[dogs["sex"]=="F"]["height"].hist(alpha=0.7)
dogs[dogs["sex"]=="M"]["height"].hist(alpha=0.7)
plt.legend(["F","M"])
plt.show()
```
## x and y axis labels, plot title, enable grid
```
plt.xlabel('...')
plt.ylabel('...')
plt.title('...')
plt.grid(True)
```
## explicit tick values and labels
```
y_val_arr = [1000, 10000, 100000] # this specific example should probably use plt.xscale('log') as well
y_label_arr = ['1k', '10k', '100k']
plt.yticks(y_val_arr, y_label_arr)
```
## barplot
```
avg_weight_by_breed=dog_pack.groupby("breed")["weight_kg"].mean()
avg_weight_by_breed.plot(kind="bar", title="...")
```




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
## Data frame from csv
```
import pandas as pd
cars = pd.read_csv('cars.csv') # Optional: specify which column should be used as index : index_col = 0 
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

