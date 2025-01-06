### Libraries
require(loomR) # For handling Loom files

### Open the existing Loom in read-only mode and import datasets
data.loom <- connect(filename = "/data/asap2/users/1/87ru1p/normalization/1742/output.loom", mode = "r")
data.parsed <- t(data.loom[["/matrix"]][, ]) # t() because loomR returns the t() of the correct matrix we want
data.loom$close_all()
