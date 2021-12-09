DATA CAMP!

# import seaborn, matplotlib and pandas
```
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
```

------------------------------------------------------------------------------------------------------------------------------------------------------
# Some simple plots : scatterplot, countplot + using hue

# scatterplot
```
height = [...]
weight = [...]
sns.scatterplot(x=height,y=weight)
plt.show()
```

# countplot
```
gender = [...]
df=pd.read_csv(...)
# With array
sns.countplot(x=gender) # use y=... instead to lay the bars down
# With DF
sns.countplot(x='gender', data=df) # use y=... instead to lay the bars down
plt.show()
```


# Add color with hue
# scatterplot example
```
hue_colors = { "Yes" : "black", "No": "red"}
sns.scatterplot(
	x="total_bill", # which df column to use for X values
	y="tip", # which df column to use for T values
	data=tips, # which df to use
	hue="smoker", # which column to use to determine color
	hue_order=["Yes","No"], # how to order the values for color 
	palette=hue_colors # specify which color column values should correspond to which colors
)
```

# countplot example
```
sns.countplot(x="smoker", data=tips, hue="sex")
plt.show()
```

------------------------------------------------------------------------------------------------------------------------------------------------------
# Relational plots and subplots - relationship between two quantitative variables in several groups using subplots rather than hue

# Scatter plot using relplot
```
sns.relplot(
	x="total_bill", # which df column to use for X values
	y="tip", # which df column to use for T values
	data=tips, # which df to use
	kind="scatter", # which type of plot to use
	col="smoker", # will create a separate sublot as columns for each unique value in "smoker" column , NOTE: use row="..." to arrange in rows
	row="meal_time", # will plot different meal times (lunch,dinner..) as rows
	col_wrap=2, # this would allow a maximum 2 columns per row, forcing the other ones to the next row
	col_order=[...] # provide array of values if custom ordering is needed
)
```

# Use col / row in replot() to create subgroups that are displayed as separate subplots

# Customize scatter plots (can be used with both scatterplot and relplot)
```
size="size" - will change the size of the data point depending on the value of the "size" data frame column value
hue="size" - will color the data point depending on the value of the "size" data frame column value
--
hue="smoker" - will color the data point depending on the value of the "smoker" data frame column value
style="smoker" - will change the point style of the data point depending on the value of the "smoker" data frame column value
--
alpha=0.4 (between 0 and 1) - changes the transparency
```


# Line plot using relplot (usually used to track a numeric value over time)
```
sns.relplot(
	x="hour", # which df column to use for X values
	y="NO_2_MEAN", # which df column to use for T values
	data=air_df_mean, # which df to use
	kind="line", # which type of plot to use,
	hue="location", # add separate colored lines for each "location" value
	style="location", # changes the line style based on the "location" value
	markers=True, # adds markers, based on values suplied to the style argument
	dashes=False # keep the style of the line from changing and stay default (useful if we only want to vary the markers)
)
# ci="sd" - sets the confidence interval to standard deviation in case the data has multiple observations for the same time value
# ci=None - if we don't want the confidence interval in case the data has multiple observations for the same time value
```

------------------------------------------------------------------------------------------------------------------------------------------------------

# Categorical plots - distribution of a quantative variable within categories defined by a categorical variable

# count plots, bar plots, box plots, point plots - using catplot()
```
sns.catplot( x="...", data=..., kind="count", order=[...,...,...] ) # simple count plot
sns.catplot( x="...", y="...", data=..., kind="bar", order=[...,...,...] ) # simple bar plot
sns.catplot( x="...", y="...", data=..., kind="box", order=[...,...,...]) # simple box plot

# sym="" - remove outliers from the boxplot
# whis=2.0 - change whiskers of the boxplot from default 1.5*IRQ to 2.0*IQR
# whis=[5,95] - change whiskers of the boxplot from default 1.5*IRQ to 5th percentile and 95 percentile
# whis=[0,100] - change whiskers of the boxplot from default 1.5*IRQ to min and max values
```

# point plot: 
# categorical variable on X axis, quantitative mean on the Y axis + confidence intervals
```
sns.catplot( x="...", y="...", data=..., kind="point", hue="...", order=[...,...,...] ) 
# (...,join=False) to remove the joining lines
# from numpy import median + (...,estimator=median) for using median instead of mean
# (..., capsize=0.2) for adding caps to the confidence intervals
# ci=None if we don't want confidence intervals
```

------------------------------------------

# Customizing plots - style and color customizing
# Presets : white, dark, whitegrid, darkgrid, ticks
```
sns.set_style("whitegrid") # To set the style
```
# Diverging color palette Examples: "RdBu", "PRGn", "RdBu_r", "PRGn_r"
# Sequential color palette examples : "Greys", "Blues", "PuRd", "GnBu"
```
sns.set_palette("RdBu") # To set a color palette
sns.set_palette(["#39A7D0","#36ADA4"]) # To set a custom color palette ( array member for each column )
```
# Scale/context examples: "paper", "notebook", "talk", "poster"
```
sns.set_context("paper") # Set scale/context
```

------------------------------------------

# Plot titles and axis labels
# FacetGrid vs. AxesSubplot objects - A FacetGrid cotains one or more AxesSubplot objects
```
g = sns.scatterplot(....) # create a visalization
type(g) # returns the object type
```
# Some visualizations support FacetGrids, some don't
# Some examples that do: relplot(), catplot()
# Some examples that don't: scatterplot(), countplot()

# Adding a title to a FacetGrid
```
g = sns.catplot(...)
g.fig.suptitle("My title"[,y=1.03]) # y - optional title height parameter
```

# Adding a title to a AxesSubplot
```
g = sns.boxplot(...)
g.set_title("My title"[,y=1.03]) # y - optional title height parameter
```

# Adding titles to FacetGrid subplots
```
g = sns.catplot(..., col="Group")
g.fig.suptitle("My title"[,y=1.03]) # alters the overall title
g.set_titles("This is {col_name}") # dynamically set subplot titles
```

# Adding axis labels
```
g.set( xlabel="...", ylabel="..." )
```

# Rotating tick labels
```
plt.xticks(rotation=90)
```
