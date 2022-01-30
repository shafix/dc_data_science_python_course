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
### scatterplot example
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

### countplot example
```
sns.countplot(x="smoker", data=tips, hue="sex")
plt.show()
```

------------------------------------------------------------------------------------------------------------------------------------------------------
# Relational plots and subplots
relationship between two quantitative variables in several groups using subplots rather than hue

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
Note: Use col / row in replot() to create subgroups that are displayed as separate subplots

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

# Categorical plots
Distribution of a quantative variable within categories defined by a categorical variable

## count plots, bar plots, box plots, point plots - using catplot()
```
sns.catplot( x="...", data=..., kind="count", order=[...,...,...] ) # simple count plot
sns.catplot( x="...", y="...", data=..., kind="bar", order=[...,...,...] ) # simple bar plot
sns.catplot( x="...", y="...", data=..., kind="box", order=[...,...,...]) # simple box plot

# sym="" - remove outliers from the boxplot
# whis=2.0 - change whiskers of the boxplot from default 1.5*IRQ to 2.0*IQR
# whis=[5,95] - change whiskers of the boxplot from default 1.5*IRQ to 5th percentile and 95 percentile
# whis=[0,100] - change whiskers of the boxplot from default 1.5*IRQ to min and max values
```

## point plot: categorical variable on X axis, quantitative mean on the Y axis + confidence intervals
```
sns.catplot( x="...", y="...", data=..., kind="point", hue="...", order=[...,...,...] ) 
# (...,join=False) to remove the joining lines
# from numpy import median + (...,estimator=median) for using median instead of mean
# (..., capsize=0.2) for adding caps to the confidence intervals
# ci=None if we don't want confidence intervals
```

------------------------------------------

# Customizing plots - style and color customizing
Presets : white, dark, whitegrid, darkgrid, ticks
```
sns.set_style("whitegrid") # To set the style
```
Diverging color palette Examples: "RdBu", "PRGn", "RdBu_r", "PRGn_r"
Sequential color palette examples : "Greys", "Blues", "PuRd", "GnBu"
```
sns.set_palette("RdBu") # To set a color palette
sns.set_palette(["#39A7D0","#36ADA4"]) # To set a custom color palette ( array member for each column )
```
Scale/context examples: "paper", "notebook", "talk", "poster"
```
sns.set_context("paper") # Set scale/context
```

------------------------------------------

# Plot titles and axis labels
FacetGrid vs. AxesSubplot objects - A FacetGrid cotains one or more AxesSubplot objects
```
g = sns.scatterplot(....) # create a visalization
type(g) # returns the object type
```
Some visualizations support FacetGrids, some don't
Some examples that do: relplot(), catplot()
Some examples that don't: scatterplot(), countplot()

### Adding a title to a FacetGrid
```
g = sns.catplot(...)
g.fig.suptitle("My title"[,y=1.03]) # y - optional title height parameter
```

### Adding a title to a AxesSubplot
```
g = sns.boxplot(...)
g.set_title("My title"[,y=1.03]) # y - optional title height parameter
```

### Adding titles to FacetGrid subplots
```
g = sns.catplot(..., col="Group")
g.fig.suptitle("My title"[,y=1.03]) # alters the overall title
g.set_titles("This is {col_name}") # dynamically set subplot titles
```

### Adding axis labels
```
g.set( xlabel="...", ylabel="..." )
```

### Rotating tick labels
```
plt.xticks(rotation=90)
```

### Pandas histogram vs seaborn histogram
```
df['Award_Amount'].plot.hist()
sns.distplot(df['Award_Amount'])
```

# Seaborn distribution plot / distplot ( one variable )
```
sns.distplot(df['COLUMN_NAME'])
# bins=10 - nr of bins
# kde=False - disables KDE
# hist=False - Removes the columns
# rug=True - creates a "rug"
# kde_kws={'shade':True} - customize KDE by passing arguments/keywords to the underlying kdeplot function
```

# Linear regression lines / plots ( two variables ) / regplot / scatterplot / lmplot
```
# Note that regplot and lmplot are similar, but lmplot is more powerful/robust
# Faceting - supplying "hue" or "col"/"row" to plot multiple graphs. 
sns.regplot(x='COLUMN1', y='COLUMN2', data=df)
sns.lmplot(x='COLUMN1', y='COLUMN2', data=df)
# hue='COLUMN3' - add multiple plots distinguished by color, based on suplied column
# col='COLUMN3' - add multiple plots, side by side, based on suplied color, can also use "row" instead
```

# Visualization styling / aesthetics / style
```
sns.set() # initilizes the default seaborn theme for a plot
sns.set_style(...) # sets a specific style , for example "white", "dark", "whitegrid", "darkgrid", "ticks"
sns.despine(...) # removes the "spines" of the plot, left=True, right=True, etc..
```

# Adjusting plot colors
```
sns.set(color_codes=True)
sns.distplot(df['Award_Amount'], color='g') # apply green color

# Cycle through various palettes
for p in sns.palettes.SEABORN_PALETTES:
	sns.set_palette(p)
	sns.palplot(sns.color_palette()) # displays palette
	sns.distplot(df['Tuition']) 
	plt.show()
# circular colors = when the data is not ordered, example sns.color_palette("Paired",12)
# sequential colors = when the data has a consistent range from low to high, for example sns.color_palette("Blues",12)
# diverging colors = when both the low and high values are interesting, for example sns.color_palette("BrBG",12)
```

# Customizing seaborn plots with matplotlib
```
# Pass "Axes" to seaborn functions
fig, ax= plt.subplots() # create figure with axes
sns.distplot(df['Tuition'], ax=ax) # pass axes to seaborn plot
ax.set(xlabel="...", ylabel="...", xlim=(0,1000), title="...") # set xlabel, ylabel, x limit and title of the plot

# Combining plots
fig, (ax0, ax1) = plot.subplots( nrows=1, ncols=2, sharey=True, figsize=(7,4) )
sns.distplot(df['Tuition'], ax=ax0)
sns.distplot(df.query[' State="MN" ']['Tuition'], ax=ax1)
ax1.set( xlabel=""Tuition (MN)", xlim(0,70000) )
ax1.axvline( x=20000, label='My Budget', linestyle='--' )
ax1.legend()
```

# Categorical plot types
stripplot and swarmplot - shows all individual observations
boxplot, violinplot, lvplot - abstract representation of the categorical data
barplot, pointplot, countplot - statistical estimates
```
sns.stripplot( data=df, y='...', x='...', jitter=True )
sns.swarmplot( data=df, y='...', x='...', hue='...' ) # not too good for large data sets

sns.boxplot( data=df, y='...', x='...' )
sns.violinplot( data=df, y='...', x='...' )
sns.lvplot( data=df, y='...', x='...' ) # letter value plot, quick to render, scales well, hybrid between box and violin plots

sns.barplot( data=df, y='...', x='...', hue='...' )
sns.pointplot( data=df, y='...', x='...', hue='...' )
sns.countplot( data=df, y='...', x='...', hue='...' )
```

# Regression plots
regression plot, residual plot

```
sns.regplot(x='COLUMN1', y='COLUMN2', data=df, marker='+') # regression plot
sns.residplot(x='COLUMN1', y='COLUMN2', data=df) # residual plot

sns.regplot(x='COLUMN1', y='COLUMN2', data=df, order=2) # polynomial regression plot
sns.residplot(x='COLUMN1', y='COLUMN2', data=df , order=2) # polynomial residual plot

sns.regplot(x='month', y='total_rentals', data=df, order=2, x_jitter=.1) # categorical regression plot with jitter
sns.regplot(x='month', y='total_rentals', data=df, order=2, x_estimator=np.mean) # categorical regression plot with estimator 

sns.regplot(x='temp', y='total_rentals', data=df, x_bins=4) # regression plot with bins
```

# Matrix plots using heatmap and pandas' crosstab and correlation functions
```
df_crosstabbed = pd.crosstab(df['month'], df['weekday']), values=df['total_rentals'], aggfunc='mean').round(0)
sns.heatmap(df_crosstabbed, annot=True, fmt='d', cmap='YlGnBu', cbar=False, linewidths=.5, center=df_crosstabbed.loc[9,6]) 
# annot=True - turns off annotations
# fmt = 'd' - formats the results as integers
# cmap = 'YlGnBu' - custom color map of yellow, green, blue
# cbar = False - remove color bar
# linewidts = .5 - adds .5 spacing between cells
# center=df_crosstabbed.loc[9,6]) - centering the heatmap on a certain area of the crosstabbed data, concentrates the color scheme more

df_corr = df.corr()
sns.heatmap(df_corr) - shows correlation between variables
```

# Combining plots into larger vizualizations using facting with FacetGrid, factorplot and lmplot - data aware plots
Note that creating these plots requires the data to be in "tidy format" - single observation per row, variables as columns
factorplot plots categorical data on a FacetGrid, simpler to use than sns.FacetGrid()
lmplot plots scatter and regression plots on a FacetGrid, simpler to use than sns.FacetGrid()
```
# Boxplot facetgrid example using sns.FacetGrid()
g = sns.FacetGrid( df, col='HIGHDEG' )
g.map( sns.boxplot, 'Tuition', order=['1','2','3','4'] )

# Pointplot example using sns.FacetGrid()
g2 = sns.FacetGrid(df,  ow="Degree_Type", row_order=['Graduate', 'Bachelors', 'Associates', 'Certificate'])
g2.map(sns.pointplot, 'SAT_AVG_ALL')

# Boxplot facetgrid example using sns.factorplot()
sns.factorplot(x="Tuition", data=df, col='HIGHDEG', kind='box')

# Pointplot example using sns.factorplot()
sns.factorplot(data=df, x='SAT_AVG_ALL', kind='point', row='Degree_Type', row_order=['Graduate', 'Bachelors', 'Associates', 'Certificate'])

# scatterplot facetgrid example using sns.FacetGrid()
g = sns.FacetGrid( df, col='HIGHDEG' )
g.map( plt.scatter, 'Tuition', 'SAT_AVG_ALL' )

# scatterplot facetgrid example using lmplot()
sns.lmplot( data=df, x='Tuition', y='SAT_AVG_ALL', col='HIGHDEG', fit_reg=False ) # fit_reg=False disables regression lines
sns.lmplot( data=df, x='Tuition', y='SAT_AVG_ALL', col='HIGHDEG', row='REGION' )
sns.lmplot(data=df, x='SAT_AVG_ALL', y='Tuition', col="Ownership", row='Degree_Type', row_order=['Graduate', 'Bachelors'], hue='WOMENONLY', col_order=inst_ord)
```

# PairGrid and pairplot - data aware plots - useful for looking at relationships between pairs of variables
```
# PairGrid examples
g = sns.PairGrid( df, vars=["col1","col2"] )
g = g.map( plt.scatter ) # scatter plots only
or
g = g.map_diag( plt.hist ) # histogram for the diag
g = g.map_offdiag( plt.scatter ) # scatter for the off-diag

# pairplot examples
sns.pairplot(df, vars=["co1","co2"], kind='reg', diag_kind='hist')
sns.pairplot(df.query('BEDROOMS<3'), vars=["co1","co2","col3"], hue='BEDROOMS', palette='husl', plot_kws={'alpha': 0.5})
sns.pairplot(data=df, x_vars=["fatal_collisions_speeding", "fatal_collisions_alc"], y_vars=['premiums', 'insurance_losses'], kind='scatter',hue='Region',palette='husl')
```

# JointGrid and jointplot - data aware grid that compares the distribution of data between two variables
```
# JointGrid example
g = sns.JointGrid( df, x='col1', y='col2' )
g = g.plot( sns.regplot, sns.distplot )
or
g = g.plot_joint(sns.kdeplot)
g = g.plot_marginals(sns.kdeplot, shade=True)
g = g.annotate(stats.pearsonr)

# jointplot example
sns.jointplot( data=df, x='col1', y='col2', kind='hex' )

# more complex jointplot example
g = ( sns
	.jointplot(
		x='Tuition',
		y='ADM_RATE_ALL',
		kind='scatter',
		xlimit=(0, 25000),
		marginal_kws=dict(
			bins=15,
			rug=True
		),
		data=df.query(' UG < 2500 & Owndership == "Public" ')
	)
	.plot_joint(sns.kdeplot)
)
```

# Plot usage:
Univariate distribution - mainly use distplot, rugplot and kdeplot are also possible alternatives
Regression analysis - lmplot works great for regression analysis and supports faceting. Also scatterplot and regplot. Helps determine linear relationships between data.
Categorical - boxplot, violin plot.


