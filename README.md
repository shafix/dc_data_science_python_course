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
## pyplot interface
```
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
# figure - object, container, holds everything that we see on the page
# axis - object, part of the page that holds the data
```
## Line plot
```
ax.plot(
	DS_COL1, # ['Mon','Tue']
	DS_COL2, # [5, 12]
	marker="o", # Marker style
	linestyle="--", # Line style, use none to remove line
	color="r" # red
)

ax.plot( seattle_weather["MONTH"], seattle_weather["MONTHLY_AVERAGE_TEMP"] ) # Line plot by default
ax.plot( texas_weather["MONTH"], texas_weather["MONTHLY_AVERAGE_TEMP"] ) # Add another plot on the same axis, both will be shown
plt.show()
```
## Axis label setting , tittle setting
```
ax.set_xlabel("bla bla")
ax.set_ylabel("bla bla")
ax.set_title("bla bla")
```
## Small multiples / subplots
```
fig, ax = plt.subplots(3, 2) # creates a figure with 3 rows of subplots and 2 columns
print(ax.shape) # [3,2]
ax[0,0].plot(...)
fig, ax = plt.subplots(2, 1) # creates a figure with 2 rows
print(ax.shape) # [2]
ax[1].plot(...)
fig, ax = plt.subplots(2, 1, sharey=True) # Makes both of the subplots share the same Y values
```


## Timeseries!
```
# Import CSV with date field as index:
climate_change = pd.read_csv("climate_change.csv", parse_dates=["date"], index_col="date")

# Plot using the date index:
ax.plot( climate_change.index, climate_change["relative_temp"] )

# Zoom in using date index:
seventies = climate_change["1970-01-01":"1979-12-31"]
ax.plot(seventies.index, seventies["co2"])
```
## Using twin axes:
```
fig, ax = plt.subplots()
ax.plot(..., color='blue') # Ex: y data meassured in the 1000s
ax.set_ylabel(..., color='blue')
ax.tick_params('y', color='blue')
ax2 = ax.twinx() # creates the 2nd y axes
ax2.plot(..., color='red') # Ex: y data meassured in the 10s
ax2.set_ylabel(..., color='red')
ax.tick_params('y', color='red')
plt.show()
```
## Custom function for plotting several time series on top of each other with different y scales and colors
```
def plot_timeseries( axes, x, y, color, xlabel, ylabel ):
	axes.plot(x, y, color=color)
	axes.set_xlabel(xlabel)
	axes.set_ylabel(ylabel, color=color)
	axes.tick_params('y', colors=color)
# Usage
fig, ax = plt.subplots()
plot_timeseries(ax, XDATA, YDATA1, 'blue', XLABEL, YLABEL1)
ax2 = ax.twinx() # creates the 2nd y axes
plot_timeseries(ax2, XDATA, YDATA2, 'red', XLABEL, YLABEL1)
plt.show()
```
## Annotating part of the plot
```
ax.annotate(
	">1 degree", 
	xy=[pd.Timestamp("2015-10-06"), 1], 
	xytext=(pd.Timestamp("2008-10-06"), -0.2),
	arrowprops={"arrowstyle":"->", "color":"gray"} 
)
```
## Barcharts - Quantitative comparisons
```
df=pd.read_csv('xxx.csv', index_col=0) # use 1st column as index when importing csv
fig, ax = plt.subplots() # create figure and axes
ax.bar(medals.index, medals["Gold"], label="Gold") # create bar with X (categories) and Y (numbers) values suplied
ax.bar(medals.index, medals["Silver"], bottom=medals["Gold"], label="Silver") # stack anoter bar on top of the previous one
ax.bar(medals.index, medals["Bronze"], bottom=medals["Gold"] + medals["Silver"], label="Bronze") # ... and another one!
ax.set_xticklabels(medals.index, rotation=90) # use X (categories) as X axes labels and rotate them by 90degrees
ax.set_ylabel("Number of medals") # Set Y label
ax.legend() # Add legend
plw.show()
```
## Histograms - Quantitative distributions
```
fig, ax = plt.subplots()
ax.hist(mens_rowing["Height"], label="Rowing", bins=5, histtype="step") # can specify bin limits bins=[12,22,55,66]...
ax.hist(mens_gymnastics["Height"], label="Gymnastics", bins=5, histtype="step")
ax.set_xlabel("Height in CM")
ax.set_ylabel("# of observations")
ax.legend()
plt.show()
```
## Statistical plotting : error bars 
```
# Bar plot with means + standard deviations as error bars
fig, ax = plt.subplots()
ax.bar(	"Rowing",
		mens_rowing["Height"].mean(),
		yerr=mens_rowing["Height"].std() )
ax.bar(	"Gymnastics",
		mens_gymnastics["Height"].mean(),
		yerr=mens_gymnastics["Height"].std() )
ax.set_ylabel("Height (cm)")
plt.show()


# Line chart with error lines
fig, ax = plt.subplots()
ax.errorbar( seattle_weather["MONTH"], seattle_weather["MLY-TAVG-NORMAL"], yerr=seattle_weather["MLY-TAVG-STDDEV"] )
ax.errorbar( austin_weather["MONTH"],  austin_weather["MLY-TAVG-NORMAL"],  yerr=austin_weather["MLY-TAVG-STDDEV"]  )
ax.set_ylabel("Temperature (Fahrenheit)")
plt.show()
```

## Boxplot
```
fig, ax = plt.subplots()
ax.boxplot( [ mens_rowing["Height"], mens_gymnastics["Height"] ] ) 
ax.set_xticklabels(["Rowing","Gymnastics"])
ax.set_ylabel("Height (cm)")
plt.show()
```
# Scatterplot
```
fig, ax = plt.subplots()
# Note: can use the attribute "c=..." to pass,for example, a time/date column to color the dots according to the time (creates a nice gradient)
ax.scatter(climate_change1["co2"], climate_change1["relative_temp"], color="red", label="cc1")
ax.scatter(climate_change2["co2"], climate_change2["relative_temp"], color="blue", label="cc2")
ax.set_xlabel("co2")
ax.set_ylabel("temp")
ax.legend()
plt.show()
```
## Changing the style of the figure
```
plt.style.use("ggplot")
plt.style.use("default")
```
## Saving the figure to a file
```
fig.savefig("filename.png") # can use quality=X [1-100] - compression ; can use dpi=X [300] - resolution ; png, svg, jpg - file format
```
## Set the size of the figure
```
fig.set_size_inches([3,5])
```
## Automating figure creation
```
fig, ax = plt.subplots()
sports=summer_2016_medals["Sport"].unique() # returns an array/series of unique sport categories
for sport in sports:
	sport_df = summer_2016_medals[summer_2016_medals["Sport"] == sport ]
	ax.bar( sport, sport_df["Height"].mean(), yerr=sport_df["Height"].std() )
ax.set_ylabel("Height (cm)") # Set Y label
ax.set_xticklabels(sports, rotation=90)
plt.show()
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



