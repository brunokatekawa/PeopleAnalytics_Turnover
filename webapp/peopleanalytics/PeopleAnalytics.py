import pickle
import inflection
import pandas as pd
import numpy as np

class PeopleAnalytics(object):
    def __init__(self):
        self.home_path = ''

        # loads the rescaling
        self.numerical_vars_scaler = pickle.load(open(self.home_path + 'parameter/numerical_vars_scaler.pkl', 'rb'))

        # loads the encoder
        self.categorical_vars_scaler = pickle.load(open(self.home_path + 'parameter/categorical_vars_scaler.pkl', 'rb'))


    def data_cleaning(self, df1):
        # 2.2 RENAMING COLUMNS

        # stores the old column names
        cols_old = ['Age', 'BusinessTravel', 'DailyRate', 'Department',
                    'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
                    'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
                    'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
                    'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
                    'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
                    'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
                    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
                    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                    'YearsWithCurrManager']

        # snake case
        snakecase = lambda x: inflection.underscore(x)

        # creates new columns from old columns in snakecase
        cols_new = list(map(snakecase, cols_old))

        # renames the old columns
        df1.columns = cols_new

        return df1


    def feature_engineering(self, df2):
        # 3.3 COLUMN FILTERING
        # cols to drop
        cols_drop = ['over18', 'standard_hours', 'employee_count', 'employee_number']

        # drops the columns
        df2 = df2.drop(cols_drop, axis=1)

        return df2


    def data_preparation(self, df3):
        # Label Encoding
        # over_time
        df3['over_time'] = df3['over_time'].apply(lambda x: 1 if x == 'Yes' else 0)

        # RESCALING
        # selects only numerical data types variables
        df_numerical_vars = df3.select_dtypes(include=['int64'])

        # scales numerical vars
        scaled_numerical = self.numerical_vars_scaler.fit_transform(df_numerical_vars)

        # gets the Data Frame version of numerical scaled for later manipulation
        df_scaled_numerical = pd.DataFrame(scaled_numerical)

        # renaming the columns of result Data Frame
        df_scaled_numerical.columns = df_numerical_vars.columns


        # ENCODING
        # selects only categorical data types variables
        df_categorical_vars = df3.select_dtypes(include=['object'])

        # One Hot Encoding
        encoded_categorical = self.categorical_vars_scaler.fit_transform(df_categorical_vars.drop('business_travel', axis=1)).toarray()

        # convert do DataFrame
        df_encoded_categorical = pd.DataFrame(encoded_categorical)

        
        # Ordinal Encoding - as there is an order
        # explicitly dictates the encoding codes
        assortment_dict = {'Non-Travel': 1, 'Travel_Rarely': 2, 'Travel_Frequently': 3}

        # extracts the column
        business_travel = df_categorical_vars['business_travel']

        # maps the values from the dict and converts to DataFrame for later manipulation
        df_business_travel = pd.DataFrame(business_travel.map(assortment_dict))


        # joins all categorical
        df_encoded_all_categorical = pd.concat([df_business_travel, df_encoded_categorical], axis=1)

        # joins scaled and encoded explain vars
        df3 = pd.concat([df_scaled_numerical, df_encoded_all_categorical], axis=1)

        return df3

    def get_prediction(self, model, original_data, test_data):
        # predicts
        pred = model.predict(test_data.values)

        employee_number = original_data['employee_number'].reset_index().drop('index', axis=1)

        # creates new column for comparison
        test_data['employee_number'] = employee_number['employee_number']

        # creates y_pred column
        test_data['y_pred'] = pred

        # filters the ones who tend to leave 
        df_tend_to_leave = test_data[(test_data['y_pred'] == 1)]

        # filters original data set to check more info about the employee
        filter_employees = original_data['employee_number'].isin(set(df_tend_to_leave['employee_number']))
        df_to_contact = original_data[filter_employees]

        return df_to_contact
