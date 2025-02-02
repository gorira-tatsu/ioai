import pandas as pd

# Read the predictions and round them to 0 or 1
df_submission = pd.read_csv("./submission.csv")
df_submission['Exited'] = df_submission['Exited'].round().astype(int)

# Read the test file to get the id column
df_test = pd.read_csv("../data/test.csv")
df_id = df_test[['id']]

# Concat the id column with the rounded predictions
df_result = pd.concat([df_id, df_submission], axis=1)

# Output the result to a new CSV file
df_result.to_csv("forsubmission_out.csv", index=False)
