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
    sys.exit()
            
# Check arguments
if(len(sys.argv) < 3): error_json("Some arguments are MISSING. Stopping...", oJSON)
#
# Imports
import scanpy as sp
#
# Open Loom file in reading mode
adata = sp.read(iFile)
#
# Write the file (will write in the new format)
sp.write(oFile, adata)
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
