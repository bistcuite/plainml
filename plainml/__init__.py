import joblib
def save_model(self,obj,name='model.pkl'):
    joblib.dump(obj, name)

def load_model(self,name='model.pkl'):
    return joblib.load(name)