from plainml import load_model

model = load_model(file_name='iris_knn.pkl')
print(model.predict([[3.0, 3.0, 3.3, 4.25]]))