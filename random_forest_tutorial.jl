
using MLDataUtils
using DecisionTree
using MLBase
include("data.jl")

# note: code for this file is taken largely from the tutorial
# https://www.linkedin.com/pulse/using-julia-random-forests-xgboost-diagnose-breast-cancer-mike-gold/?published=t


dataset = get_data()

# println(first(dataset, 1))
println(describe(dataset))
println(size(dataset))
println(dataset[1:3, [:ID, Symbol("B/M")]])

println("Malignant: $(size(dataset[dataset[Symbol("B/M")].=="M", :])).")
println("Benign: $(size(dataset[dataset[Symbol("B/M")].=="B", :])).")

x = convert(Array, dataset[:,3:32])
y = convert(Array,  map(element -> element == "B" ? 1 : 2,
                        dataset[Symbol("B/M")]))

# randomize the cancer data rows so when we split the data, we don't only pull one cancer classification
# also we need to transpose the x data,
#because it needs to align properly with the output dimensions
Xs, Ys = shuffleobs((transpose(x), y))
#split the data now
(X_train1, y_train1), (X_test1, y_test1) = splitobs((Xs, Ys); at = 0.67)

# transpose the x data back to its original dimensionality
x_train = Array(transpose(X_train1))
y_train = Array(y_train1)
x_test = Array(transpose(X_test1))
y_test = Array(y_test1)

model = RandomForestClassifier(n_subfeatures = 20, n_trees = 100,
                               partial_sampling = 0.7, max_depth = 7)
DecisionTree.fit!(model, x_train, y_train)

r_f_prediction = convert(Array{Int64,1}, DecisionTree.predict(model, x_test))
println("test-set error rate: $(errorrate(y_test, r_f_prediction))")

println("test-set confusion matrix: $(confusmat(2, y_test, r_f_prediction))")
