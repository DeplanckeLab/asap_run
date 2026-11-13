# Imports (except numpy, because nthreads should be defined before numpy is imported)
import sys
import time
import math
import json

# Arguments
loom_file = 'MISSING'
iAnnot = 'MISSING'
oAnnot = 'MISSING'
oJSON = 'MISSING'
n_comps = 'MISSING'
chunksize = 'MISSING'
n_threads = 'MISSING'
time_idle = 0

if len(sys.argv) >= 2: loom_file = sys.argv[1]
if len(sys.argv) >= 3: iAnnot = sys.argv[2]
if len(sys.argv) >= 4: oAnnot = sys.argv[3]
if len(sys.argv) >= 5: oJSON = sys.argv[4]
if len(sys.argv) >= 6: n_comps = int(sys.argv[5])
if len(sys.argv) >= 7: chunksize = int(sys.argv[6])
if len(sys.argv) >= 8: n_threads = sys.argv[7] # Should be str

print('PCA optimized from scanpy - List of arguments:')
print('1. Loom file:', loom_file)
print('2. Metadata to read:', iAnnot)
print('3. Metadata to write:', oAnnot)
print('4. Output JSON file:', oJSON)
print('5. Number of PCs to compute:', n_comps)
print('6. Chunk size i.e. nb of cells to compute for every chunk:', chunksize)
print('7. Nb of threads to use for parallelization:', n_threads)

# Functions
# Handling errors
def error_json(message, jsonfile):
    print(message, file=sys.stderr)
    if(jsonfile != 'MISSING'):
        data_json = {}
        data_json['displayed_error'] = message
        with open(jsonfile, 'w') as outfile:
            json.dump(data_json, outfile)
    sys.exit(1)

# Open the Loom file while handling potential locking
def open_with_lock(loomfile, mode):
    global time_idle, oJSON
    while True:
        try:
            return h5py.File(loomfile, mode)
        except Exception as e:
            if not "unable to lock file" in format(e): error_json("Error opening Loom file:" + format(e), oJSON)
        print("Sleeping 1sec for file lock....")
        time_idle += 1
        time.sleep(1)
            
# Check arguments
if(len(sys.argv) < 8): error_json("Some arguments are MISSING. Stopping...", oJSON)

# Imports
import os
os.environ["OMP_NUM_THREADS"] = n_threads
os.environ["OPENBLAS_NUM_THREADS"] = n_threads
os.environ["MKL_NUM_THREADS"] = n_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads
os.environ["NUMEXPR_NUM_THREADS"] = n_threads
import h5py
import numpy
from sklearn.decomposition import IncrementalPCA

# Open Loom file in reading mode
f = open_with_lock(loom_file, 'r')

# Open dataset
if not iAnnot in f: error_json("This dataset is not present in the Loom file", oJSON)
m = f[iAnnot]
if hasattr(m, 'chunks'): chunksize = math.ceil(chunksize / m.chunks[1]) * m.chunks[1]

# Test if we can compute this number of PCs
if(n_comps > m.shape[0]): error_json("You cannot perform a PCA of more than " + str(m.shape[0]) + " components.", oJSON)
if(n_comps > m.shape[1]): error_json("You cannot perform a PCA of more than " + str(m.shape[1]) + " components.", oJSON)

# Prepare PCA objects
X_pca = numpy.zeros((m.shape[1], n_comps), m.dtype) # Shape = (nbcells, 50) # NEED TO TRANSPOSE THE MATRIX
pca_ = IncrementalPCA(n_components=n_comps)

# Run chunks of all genes x n cells
totalChunks = math.ceil(m.shape[1] / chunksize)
# gen_batches covers every cell (range(1, n, chunksize) skipped the last one when n % chunksize == 1).
# min_batch_size: partial_fit needs >= n_comps cells per batch, so a short tail is merged into the previous batch.
from sklearn.utils import gen_batches
fit_batches = list(gen_batches(m.shape[1], max(chunksize, n_comps), min_batch_size=n_comps))
for currentChunk, sl in enumerate(fit_batches, start=1):
    print ("Processing chunk", currentChunk, "(",currentChunk, "/", len(fit_batches), ":", '%.1f'%((currentChunk / len(fit_batches)) * 100),"%)", time.ctime()) 
    chunk = m[:,sl].transpose() # Lines = genes / Cols = cells
    pca_.partial_fit(chunk)

# Do it again, this time to generate the final PCs
for currentChunk, sl in enumerate(gen_batches(m.shape[1], chunksize), start=1):
    print ("Processing chunk", currentChunk, "(",currentChunk, "/", totalChunks, ":", '%.1f'%((currentChunk / totalChunks) * 100),"%)", time.ctime()) 
    chunk = m[:,sl].transpose() # Lines = genes / Cols = cells
    X_pca[sl] = pca_.transform(chunk)

# Close the Loom file (reading mode)
f.close()

# Open in writing to write output
f = open_with_lock(loom_file, 'r+')
if oAnnot in f: del f[oAnnot] # Delete output dataset if exists
f.create_dataset(oAnnot, data=X_pca, chunks=(min(X_pca.shape[0], 32), min(X_pca.shape[1], 32)), compression="gzip", compression_opts=2)
f.close()

# Prepare output.json
data_json = {}
data_json['time_idle'] = time_idle
data_json['metadata'] = [{
    'name': oAnnot,
    'on': 'CELL',
    'type': 'NUMERIC',
    'nber_cols' : X_pca.shape[0],
    'nber_rows' : X_pca.shape[1]
}]

# Write output.json file
with open(oJSON, 'w') as outfile:
    json.dump(data_json, outfile)
