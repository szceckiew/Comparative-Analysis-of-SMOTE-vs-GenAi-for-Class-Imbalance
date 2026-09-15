from imblearn.over_sampling import SMOTE

def apply_smote(x_train, y_train):
    sm = SMOTE(random_state=42)
    x_res, y_res = sm.fit_resample(x_train, y_train)
    return x_res, y_res
