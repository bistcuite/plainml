# PlainML
Painless Machine Learning Library for python based on [scikit-learn](https://scikit-learn.org/stable/).

## Install
```
pip install plainml
```

## Example
```py
from plainml import KnnModel, load_iris, train_test_split

dt = load_iris()
data = dt.data
target = dt.target

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

model = KnnModel(x_train, y_train)
model.fit()

print(model.score(x_test, y_test))
print(model.predict([[5.1, 3.5, 1.4, 0.2]]))

model.save(file_name='iris_knn.pkl')
```

## License
[MIT](LICENSE).
