### Libraries
require(loomR) # For handling Loom files

# Open Loom in writing mode
time_idle = 0
data.loom <- connect(filename = "/data/asap2/users/1/87ru1p/normalization/1742/output.loom", mode = "r")
repeat{ # Handle the lock of the file
  #isLocked = F
  #tryCatch({
    data.parsed <- t(data.loom[["/matrix"]][, ])
  #  if(data.loom$exists("/matrix")) data.loom$link_delete("/matrix") # Remove existing dimension reduction with same name
  #  data.loom[["/matrix"]] = t(data.parsed)
  #}, error = function(err) {
  #  if(grepl("unable to lock file", err$message)) isLocked = T
  #  else error.json(err$message)
  #})
  #if(!isLocked) break
  #else {
    message("Sleeping 1sec for file lock....")
    time_idle <<- time_idle + 1
    Sys.sleep(1)
  #}
}
data.loom$close_all()