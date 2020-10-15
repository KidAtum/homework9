# Lucas Weakland
# SCS 235
# Homework 9

# Imports
import pandas as pd

# initialize list of lists
data = [['Brad', 29], ['Sasmor', 50], ['Barry', 114]]

# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['Name', 'Age'])

yesNo = input('Hello there Brad.. Would you like to run this manual panda? Type yes or no.')
if yesNo.lower() == 'yes':
  print("Mmmmm, solid choice!")
  print(df)
else:
  print ("Oh okay whatever, cya.")


