from classes.data import DataOps


# define total images per candidate and test val split
COUNT = 450
SPLIT = 0.85

# create dataset
ds = DataOps(COUNT, SPLIT)
ds.create_dataset()
