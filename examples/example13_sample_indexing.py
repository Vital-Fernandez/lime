# import pandas as pd

# assign data of lists.
import pandas as pd

data1 = {'Name': ['Renault', 'Duster', 'Maruti', 'Honda City'], 'Ratings': [9.0, 8.0, 5.0, 3.0]}
df1 = pd.DataFrame(data1, index=['position1', 'position2', 'position3', 'position4'])

data2 = {'Name': ['Renault2', 'Duster2'], 'Ratings': [20, 30]}
df2 = pd.DataFrame(data2, index=['position1', 'position4'])

df_dict = {'obj1':df1, 'obj2':df2}

print(df1)
print(df2)

mDf = pd.concat(list(df_dict.values()), axis=0, keys=list(df_dict.keys()))
print(mDf)

