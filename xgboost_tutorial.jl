
using XGBoost  # note: wrapper for c++, not pure Julia implementation
include("data.jl")

# note: code for this file is taken largely from the tutorial
# https://www.linkedin.com/pulse/using-julia-random-forests-xgboost-diagnose-breast-cancer-mike-gold/?published=t


dataset = get_data()

x = convert(Array, dataset[:,3:32])
y = convert(Array,  map(element -> element == "B" ? 0 : 1, dataset[Symbol("B/M")]))

Xs, Ys = shuffleobs((transpose(x), y))
(X_train1, y_train1), (X_test1, y_test1) = splitobs((Xs, Ys); at = 0.67)

x_train = Array{Float32}(transpose(X_train1))
y_train = Array{Int32}(y_train1)
x_test = Array{Float32}(transpose(X_test1))
y_test = Array{Int32}(y_test1)

dtrain = DMatrix(x_train, label = y_train)
boost = xgboost(dtrain, 10, eta = 1, objective = "binary:logistic")
predictions = XGBoost.predict(boost, x_test)

println("predictions: $(predictions)")
