import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from correlation import data, correlation_data

st.title("Thesis")

# Display formatted text
st.markdown("**Welcome to my thesis project.** This app demonstrates ...")

st.write("1. Load Dataset")

dataset = pd.read_csv("synthetic_risk_factors_data.csv")
df_without_email = dataset.drop(columns=['Email Address'])
st.write(df_without_email)

st.write("2. Replace Yes and No to 1 and 0 respectively, incase of null value populate by calulating median")
# Replace 'Yes' and 'No' with 1 and 0 respectively
df_without_email.replace({'Yes': 1, 'No': 0}, inplace=True)
# calculate median of each column
medians = df_without_email.median()
# replace NaN values with median of respective column
preprocessed_df = df_without_email.fillna(medians)
st.write(preprocessed_df)


# Normalize weights where 'preprocessed_df' is your DataFrame containing risk factor columns that sum up to 100%
# List of risk factor columns
risk_factor_columns = ['Architecture Complexity', 'Performance Issues', 'Scalability Issues',
                       'Compatibility Issues', 'Budget Constraints', 'Schedule Constraints',
                       'Scope Constraints', 'Resource Constraints', 'Market Changes',
                       'Competition', 'Regulatory Requirements', 'Skill Gaps',
                       'Turnover', 'Team Communication Issues']

# Calculate mean values for each risk factor
mean_values = preprocessed_df[risk_factor_columns].mean()

# Display mean values before normalization
# print("Mean values before normalization:")
# print(mean_values)
st.text("3. Mean values before normalization:")
st.write(mean_values)

# Normalize risk factor weights to ensure the sum is 100%
normalized_weights = preprocessed_df[risk_factor_columns] / preprocessed_df[risk_factor_columns].sum(axis=1)[:, None] * 100

# Display normalized weights
# print("\nNormalized Risk Factor Weights:")
st.text("4. Normalized Risk Factor Weights:")
st.write(normalized_weights)

# Assuming 'df' is your DataFrame containing the risk factors columns
# Compute the sum of occurrences (sum of 1s) for each risk factor
sum_of_occurrences = preprocessed_df[['Architecture Complexity', 'Performance Issues', 'Scalability Issues',
                         'Compatibility Issues', 'Budget Constraints', 'Schedule Constraints',
                         'Scope Constraints', 'Resource Constraints', 'Market Changes',
                         'Competition', 'Regulatory Requirements', 'Skill Gaps',
                         'Turnover', 'Team Communication Issues']].sum()

total_occurrences = sum_of_occurrences.sum()
st.markdown(f"5. Compute the sum of occurrences (sum of 1s) for each risk factor :  **{total_occurrences}**.")
st.write() 

# Compute total occurrences
# total_occurrences = df.shape[0]  # Total number of rows in the DataFrame
# Calculate weights by dividing the sum of occurrences by the total number of rows
weights = ( sum_of_occurrences / total_occurrences ) * 100

# Display the calculated weights
# print(weights)
# print(weights.sum())
st.text("6. Calculate weights by dividing the sum of occurrences by the total number of rows")
st.write(weights)


# Multiply values in each column with respective weight
df_weighted = preprocessed_df.iloc[:,3:].apply(lambda x: x*weights[x.name])

# Sum weighted values across columns to get total score
preprocessed_df['Total Score'] = df_weighted.sum(axis=1)

# Normalize total score to 100%
preprocessed_df['Risk Score'] = preprocessed_df['Total Score'] / preprocessed_df['Total Score'].max() * 100

# Print resulting dataframe with risk scores
# print(preprocessed_df[['Risk Score']])
st.write("7. Caculated Risk Score")
st.write(preprocessed_df)
# st.write(preprocessed_df['Risk Score'])


# Plot Risk Score
st.text("8. Risk Score Distribution")
plt.figure(figsize=(10, 6))
plt.plot(preprocessed_df.index, preprocessed_df['Risk Score'], marker='o')
plt.title('Risk Score Distribution')
plt.xlabel('Index')
plt.ylabel('Risk Score')
st.pyplot(plt)

# Calculate the correlation matrix
# Define risk factors
risk_factors = [
    "Architecture Complexity", "Performance Issues", "Scalability Issues",
    "Compatibility Issues", "Budget Constraints", "Schedule Constraints",
    "Scope Constraints", "Resource Constraints", "Market Changes",
    "Competition", "Regulatory Requirements", "Skill Gaps", "Turnover",
    "Team Communication Issues"
]


# Define the correlation matrix
dataInput = data 

# data = {
#     "Architecture Complexity": [1.0000, 0.6950, 0.5842, 0.1926, -0.2010, 0.3112, 0.4336, 0.1997, 0.0214, 0.1221, 0.2054, 0.0074, 0.1325, 0.2715],
#     "Performance Issues": [0.6950, 1.0000, 0.6785, 0.3212, -0.1842, 0.3876, 0.5187, 0.3211, 0.1202, 0.1896, 0.3165, 0.0998, 0.6101, 0.7178],
#     "Scalability Issues": [0.5842, 0.6785, 1.0000, 0.2836, -0.0921, 0.2097, 0.3892, 0.3062, 0.1061, 0.2194, 0.3118, 0.2233, 0.4821, 0.6100],
#     "Compatibility Issues": [0.1926, 0.3212, 0.2836, 1.0000, 0.5143, 0.1103, 0.2176, 0.5214, 0.0056, 0.0954, 0.2056, 0.0014, 0.2094, 0.3102],
#     "Budget Constraints": [-0.2010, -0.1842, -0.0921, 0.5143, 1.0000, 0.3042, 0.5071, 0.8225, 0.2153, 0.1082, 0.6123, 0.2104, 0.3211, 0.4052],
#     "Schedule Constraints": [0.3112, 0.3876, 0.2097, 0.1103, 0.3042, 1.0000, 0.5039, 0.4185, 0.0999, 0.2085, 0.3112, 0.2048, 0.3215, 0.3971],
#     "Scope Constraints": [0.4336, 0.5187, 0.3892, 0.2176, 0.5071, 0.5039, 1.0000, 0.6163, 0.1058, 0.2034, 0.3985, 0.2977, 0.4108, 0.5187],
#     "Resource Constraints": [0.1997, 0.3211, 0.3062, 0.5214, 0.8225, 0.4185, 0.6163, 1.0000, 0.2075, 0.1023, 0.3057, 0.2065, 0.3140, 0.4186],
#     "Market Changes": [0.0214, 0.1202, 0.1061, 0.0056, 0.2153, 0.0999, 0.1058, 0.2075, 1.0000, 0.8064, 0.5062, 0.1986, 0.3084, 0.3989],
#     "Competition": [0.1221, 0.1896, 0.2194, 0.0954, 0.1082, 0.2085, 0.2034, 0.1023, 0.8064, 1.0000, 0.6179, 0.3020, 0.4102, 0.5025],
#     "Regulatory Requirements": [0.2054, 0.3165, 0.3118, 0.2056, 0.6123, 0.3112, 0.3985, 0.3057, 0.5062, 0.6179, 1.0000, 0.3072, 0.4084, 0.5118],
#     "Skill Gaps": [0.0074, 0.0998, 0.2233, 0.0014, 0.2104, 0.2048, 0.2977, 0.2065, 0.1986, 0.3020, 0.3072, 1.0000, 0.5119, 0.6112],
#     "Turnover": [0.1325, 0.6101, 0.4821, 0.2094, 0.3211, 0.3215, 0.4108, 0.3140, 0.3084, 0.4102, 0.4084, 0.5119, 1.0000, 0.7102],
#     "Team Communication Issues": [0.2715, 0.7178, 0.6100, 0.3102, 0.4052, 0.3971, 0.5187, 0.4186, 0.3989, 0.5025, 0.5118, 0.6112, 0.7102, 1.0000]
# }

# Create DataFrame
fdf = pd.DataFrame(dataInput, index=data.keys())

# Create DataFrame
correlation_df = pd.DataFrame(fdf, index=risk_factors)

st.write("9. Correlation Matrix (Interdependance between each Risk Factors):")

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Risk Factors Correlation Heatmap')
st.pyplot(plt)


st.write("10. Correlation Matrix (Interdependance between each Risk Factors Category):")

# # Aggregate the risk factors into their respective categories using mean
# fdf['Technical Risks'] = fdf[['Architecture Complexity', 'Performance Issues', 'Scalability Issues', 'Compatibility Issues']].mean(axis=1)
# fdf['Project Risks'] = fdf[['Schedule Constraints', 'Budget Constraints', 'Resource Constraints', 'Scope Constraints']].mean(axis=1)
# fdf['Business Risks'] = fdf[['Market Changes', 'Competition', 'Regulatory Requirements']].mean(axis=1)
# fdf['Personal Risks'] = fdf[['Skill Gaps', 'Turnover', 'Team Communication Issues']].mean(axis=1)

correlation_data_input = correlation_data
# correlation_data = {
#     'Technical Risks': [1.00, 0.6805, 0.5214, 0.3265],
#     'Project Risks': [0.6805, 1.00, 0.5825, 0.5913],
#     'Business Risks': [0.5214, 0.5825, 1.00, 0.3125],s
#     'Personal Risks': [0.3265, 0.5913, 0.3125, 1.00]
# }

# Create a DataFrame
correlation_cat_df = pd.DataFrame(correlation_data_input, index=['Technical Risks', 'Project Risks', 'Business Risks', 'Personal Risks'])

# Create a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_cat_df, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Risk Categories')
st.pyplot(plt)


# correlation_matrix = preprocessed_df[risk_factor_columns].corr()
# st.write(correlation_matrix)

# # Create a heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
# plt.title('Correlation Matrix of Risk Factors')
# st.pyplot(plt)
