# Numpy
### Numpy arrays ( 1d vectors / 2d matrices / 3d+ tensors )
python lists can have members of any data type
numpy array members are all of the same data type
tuple - created using (), similar to list just immutable
```
import numpy as np
# 1 dimensional and 2 dimensional array examples
np_arr_1d = np.array( [1,2,3,4,5] )
np_arr_2d = np.array( [[1,2,3,4,5],
					   [2,3,4,5,6]])
# Some other useful np functions
np.zeros((5,3)) # Makes a 5x3 2d array of zeros
np.random.random((5,3)) # Makes a 5x3 2d array of random 0-1 float numbers
np.arange(-3,4) # Creates a 1d evenly spaced array to fill between given start and stop values (excluding stop), sequential integers, can add step as 3rd argument

# 3d array example
array_1_2D = np.array([[1, 2], [5, 7]])
array_2_2D = np.array([[8, 9], [5, 7]])
array_3_2D = np.array([[1, 2], [5, 7]])
array_3D = np.array([array_1_2D, array_2_2D, array_3_2D])

# Array attributes and methods
# .shape - describes array, returns tuple of the length of each dimension
# .flatten() - forces the n dimension array into 1 dimension
# .reshape() - reshape array, for example 2x3 to 3x2 (must be compatable numbers)

# Creating 32bit rather than the 64bit default float array
float32_array = np.array([1.32, 5.78, 175.55], dtype=np.float32)
float32_array.dtype # dtype('float32')

# Convert type
boolean_array = np.array([[True, False], [False, False]], dtype=np.bool_)
boolean_array.astype(np.int32)
```

# Indexing and slicing arrays
```
sudoku_game[2,4] <- 3rd row, 5th column
sudoku_game[0] <- returns the whole 1st row
sudoku_game[:,3] <- all row data at 4th column
array[2:4] <- returns element from 3rd (inclusive) to 5th(exclusive)
sudoku_game[3:6,3:6] <- returns 3,4 and 5th columns of the 3,4 and 5th rows.
sudoku_game[3:6:2,3:6:2] <- returns 3 and 5th columns of the 3 and 5th rows. Step value argument.
np.sort(sudoku_game) <- sorts the data along the column axis(1) by default
np.sort(sudoku_game, axis=0) <- sorts the data along the row axis(0)
```

# Filtering arrays - masks and fancy indexing vs np.where()
```
one_to_five = np.arange(1,6)
mask = one_to_five % 2 == 0 # <- array of booleans
one_to_five[mask] # <- filters out members that are not divisible by 2

classrooom_ids_and_sizes = np.array([[1, 22], [2, 21], [3, 27], [4, 26]])
mask = classroom_ids_and_sizes[:, 1] % 2 == 0 # which clasroom sizes are divisible by 2? Creates boolean array
classrooom_ids_and_sizes[mask] # return the classes where the size is divisible by 2

# 1d array with where
np.where(classroom_ids_and_sizes[:, 1] % 2 == 0) # returns (a tuple of) array of indices

trunk_stump_diameters = np.where( tree_census[:,2]==0, tree_census[:,3], tree_census[:,2] ) # 2=trunk diam, 3=stump, this reads replace trunk diam with stump diam if trunk diam is 0, returns array of trunk diams

# 2d array with where
row_ind, column_ind = np.where(sudoku_game == 0)
np.where(sudoku_game == 0, "", sudoku_game) # returns back the same array, but replaces 0s with empty strings in this case
```

# Concatenation - adding rows or columns with np.concatenate()
by default (axis=0) - adding rows (column count must match)
by argument (axis=1) - adding columns (row count must match)
```
np.concatenate( ( array_1 , array_2 ) ) # adding rows
np.concatenate( ( array_1 , array_2 ), axis=1 ) # adding columns
```
Note: must reshape 1d array into 2d array before concatinating with a 2d array
```
array_1d = np.array([1,2,3])
column_array_2d = array_1d.reshape((3,1)) # [ [1], [2], [3] ]
row_array_2d = array_1d.reshape((1,3)) # [ [1, 2, 3] ]
```

# Deletion with np.delete()
Takes 3 arguments: array from which to delete, slice/index/array of incides which signifying which rows/columns to delete, axis on which to delete (0 being row, 1 being column)
```
np.delete( some_arr, 1, axis=0 ) # deletes 2nd row of the 2d array
np.delete( some_arr, 1, axis=1 ) # deletes 2nd column of the 2d array
```

# Combining np.delete() and np.where():
```
tree_census_no_stumps = np.delete(tree_census, 3, axis=1) # # Delete the stump diameter column from tree_census
private_block_indices = np.where(tree_census[:,1] == 313879) # # Save the indices of the trees on block 313879
tree_census_clean = np.delete( tree_census_no_stumps, private_block_indices, axis=0 ) # # Delete the rows for trees on block 313879 from tree_census_no_stumps
```

# Summarizing array data with aggregate functions : 
aggregates across all dimentions by default, can use aixs=0 for column totals (1 sum for each column) or use aixs=1 for row totals (1 sum for each row):
.sum()
.min()
.max()
.mean()
note: can also use keepdims=True to make sure the result is a 2d array

.cumsum() - returns cumulative sums

Example:
```
monthly_industry_sales = monthly_sales.sum(axis=1, keepdims=True) # Create a 2D array of total monthly sales across industries
monthly_sales_with_total = np.concatenate( (monthly_sales, monthly_industry_sales), axis=1 ) # Add this column as the last column in monthly_sales
```

# Vectorized operations - apply function to each member of the array
```
some_array + 3 # adds 3 to all members of the array
some_2d_array_1 + some_2d_array_2 # adds up the arrays on the same dimension / location
some_2d_array > 2 # checks member of the array, returns boolean aray with the same structure

len(array) > 2 # true, pyhon function!
vectorized_len = np.vectorize(len)
vecrorized_len(array) > 2 # returns array of booleans, performing operation on each member
```

# Broadcasting - stretching smaller array (example scalar) over larger one
Note that dimensions need to be compatable between the arrays - they have to be the same lenght or one of them has to be 1
Basically copying the column values into as many rows as needed or copying the row values into as many columns as necessary

# Saving and loading numpy arrays
Can be saved as .csv, .txt, .pkl, but the best is .npy
Loading a file:
```
with open('logo.npy','rb') as f: # rb=read binary mode
	logo_rgb_array = np.load(f)
	
# Unrelated to opening...	
plt.imshow(logo_rgb_array)
plt.show()
dark_logo_array = np.where(logo_rgb_array == 255, 50, logo_rgb_array) # swaps white pixels with dark grey ones
plt.imshow(dark_logo_array)
plt.show()
```
Saving a file:
```
with open('dark_logo_array.npy','wb') as f: # rb=write binary mode
	np.save(f, dark_logo_array) # will be saved as dark_logo_array.npy, overwritten if exists
```

Array acrobatics:
```
np.flip( ARR, AXIS_ARG ) # reverses the order of array elements on every axis. Can provide axis argument so that it only flips on certain exis. ex: axis=0, axis=1, axis=(0,1)
np.transpose( ) # flips axis order, but elements inside axis stay the same, columns become rows, rows become columns. Can provide axes argument to specify exact flipping. ex axes=(1,0,2)
```

Stacking and splitting:
```
red_array, green_array, blue_array = np.split( rgb_array, 3, axis=2 )
stacked_rgb = np.stack ( [red_array, green_array, blue_array], axis=2 )
```
