import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess(df, option):
    """
    This function is to cover all the preprocessing steps on the churn dataframe. It involves selecting important features, encoding categorical data, handling missing values,feature scaling and splitting the data
    """
    
    print(df.dtypes)
    #Defining the map function
    def binary_map(feature):
        return feature.map({'Yes':1, 'No':0, 'Female' : 1, 'Male':0})

    # Some of the columns have no internet service or no phone service, that can be replaced with a simple No
    df.replace('No internet service','No',inplace=True)
    df.replace('No phone service','No',inplace=True)
    
    # Encode binary categorical features
    
    if option == 'Online':
        binary_list = ['gender','SeniorCitizen' ,'Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']
        df[binary_list] = df[binary_list].apply(binary_map)
        
    elif option == 'Batch':
        binary_list = ['gender','Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']
        df[binary_list] = df[binary_list].apply(binary_map)
            
    #Encoding the other categorical categoric features with more than two categories
    col_names = ['gender','SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'OnlineSecurity', \
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', \
       'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', \
       'InternetService_DSL', 'InternetService_Fiber optic', \
       'InternetService_No', 'Contract_Month-to-month', 'Contract_One year', \
       'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)', \
       'PaymentMethod_Credit card (automatic)', \
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
    
    df = pd.get_dummies(data=df, columns = ['InternetService','Contract','PaymentMethod']).reindex(columns=col_names, fill_value=0)

    # We remove these irrelevant columns : ['gender', 'PhoneService', 'MultipleLines']
    df = df.drop(['gender', 'PhoneService', 'MultipleLines'], axis=1)
        
    #feature scaling
    sc = MinMaxScaler()
    df['tenure'] = sc.fit_transform(df[['tenure']])
    df['MonthlyCharges'] = sc.fit_transform(df[['MonthlyCharges']])
    df['TotalCharges'] = sc.fit_transform(df[['TotalCharges']])
    
    
    return df



