import sys 
import  json

# Arguments
iFile = 'MISSING'
oFile = 'MISSING'
oJSON = 'MISSING'
#
if len(sys.argv) >= 2: iFile = sys.argv[1]
if len(sys.argv) >= 3: oFile = sys.argv[2]
if len(sys.argv) >= 4: oJSON = sys.argv[3]
#
print('H5ad format - List of arguments:')
print('1. Input H5ad file:', iFile)
print('2. Output H5ad file:', oFile)
print('3. Output JSON file:', oJSON)
#
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
            
# Check arguments
if(len(sys.argv) < 3): error_json("Some arguments are MISSING. Stopping...", oJSON)
#
# Imports
import scanpy as sp
import pandas as pd
import anndata
anndata.settings.allow_write_nullable_strings = True # only string columns with missing values remain nullable (see below)
#
def to_object_strings(frame):
    # pandas 3 reads text as the nullable 'str' dtype, which anndata refuses to write by default
    # (and older readers cannot read): cast index/columns/categories back to object dtype.
    if isinstance(frame.index.dtype, pd.StringDtype) and not frame.index.hasnans:
        frame.index = pd.Index(frame.index, dtype=object, name=frame.index.name)
    for col in list(frame.columns):
        s = frame[col]
        if isinstance(s.dtype, pd.CategoricalDtype) and isinstance(s.cat.categories.dtype, pd.StringDtype):
            frame[col] = s.cat.rename_categories(pd.Index(s.cat.categories, dtype=object))
        elif isinstance(s.dtype, pd.StringDtype) and not s.isna().any():
            frame[col] = s.astype(object)
#
try:
    # Open H5ad file in reading mode
    adata = sp.read(iFile)
    to_object_strings(adata.obs)
    to_object_strings(adata.var)
    if adata.raw is not None: to_object_strings(adata.raw.var)
    # Write the file (will write in the new format)
    sp.write(oFile, adata)
except Exception as e:
    error_json("Could not reformat the H5ad file: " + type(e).__name__ + ": " + str(e), oJSON)
#
# Prepare output.json
data_json = {}
data_json['detected_format'] = 'H5AD'
data_json['iFile'] = iFile
data_json['oFile'] = oFile
data_json['nber_rows'] = adata.shape[1]
data_json['nber_cols'] = adata.shape[0]
#
# Write output.json file
with open(oJSON, 'w') as outfile:
    json.dump(data_json, outfile)
