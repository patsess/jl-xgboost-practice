
using DataFrames
using CSV

# note: code for this file is taken largely from the tutorial
# https://www.linkedin.com/pulse/using-julia-random-forests-xgboost-diagnose-breast-cancer-mike-gold/?published=t


function get_data()
    dataset = CSV.read(
        download("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"),
        header = vcat(["ID", "B/M"], collect(1:30)),
        type = Dict(1 => Int64, 2=>String))
    return dataset
end
