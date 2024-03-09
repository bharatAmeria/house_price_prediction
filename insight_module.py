from statistics import LinearRegression

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler


class InsightModule:
    def __init__(self):
        pass

    def main(self):
        df['agePossession'] = df['agePossession'].replace(
            {
                'Relatively New': 'new',
                'Moderately Old': 'old',
                'New Property': 'new',
                'Old Property': 'old',
                'Under Construction': 'under construction'
            }
        )

        df['property_type'] = df['property_type'].replace({'flat': 0, 'house': 1})
        df['luxury_category'] = df['luxury_category'].replace({'Low': 0, 'Medium': 1, 'High': 2})
        new_df = pd.get_dummies(df, columns=['sector', 'agePossession'], drop_first=True)

        X = new_df.drop(columns=['price'])
        y = new_df['price']

        y_log = np.log1p(y)

        scaler = \
            StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(LinearRegression(), X_scaled, y_log, cv=kfold, scoring='r2')

        lr = LinearRegression()
        ridge = Ridge(alpha=0.0001)

        lr.fit(X_scaled, y_log)
        ridge.fit(X_scaled, y_log)

        coef_df = pd.DataFrame(ridge.coef_.reshape(1, 112), columns=X.columns).stack().reset_index().drop(
            columns=['level_0']).rename(columns={'level_1': 'feature', 0: 'coef'})

        # 1. Import necessary libraries
        import statsmodels.api as sm

        # 2. Add a constant to X
        X_with_const = sm.add_constant(X_scaled)

        # 3. Fit the model
        model = sm.OLS(y_log, X_with_const).fit()

        # 4. Obtain summary statistics
        print(model.summary())

