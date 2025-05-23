package hdf5;

import bigarrays.LongArray64;
import ch.systemsx.cisd.hdf5.HDF5DataClass;
import ch.systemsx.cisd.hdf5.IHDF5Reader;
import db.EnsemblDB;
import json.ErrorJSON;
import json.ParsingJSON;
import model.Parameters;
import tools.Utils;

public class ProgressiveReader 
{
	private IHDF5Reader reader = null;
	private String path = null;
	private int dataBlockSize = -1;
	private int indicesBlockSize = -1;
	private HDF5DataClass type = HDF5DataClass.INTEGER;
	
	public LongArray64 indptr = null; // column indexes 'indptr', required for recreating the dense matrix
	
	Block dataBlocks = null;
	Block indicesBlocks = null;
	
	private int currentDataBlockRead = -1;
	private int currentIndicesBlockRead = -1;
	
	private long nbGenes = -1;
	private long nbCells = -1;
	private int blockSizeX = -1;
	public int nbTotalBlocks = -1;
	
	public ProgressiveReader(IHDF5Reader reader, long nbGenes, long nbCells, int blockSizeX, int blockSizeY) 
	{
		this(reader, "/" + Parameters.selection, nbGenes, nbCells, blockSizeX, blockSizeY);
	}
	
	public ProgressiveReader(IHDF5Reader reader, String path, long nbGenes, long nbCells, int blockSizeX, int blockSizeY) 
	{
		this.path = path;
		this.reader = reader;
		this.blockSizeX = blockSizeX;
		this.dataBlockSize = this.reader.getDataSetInformation(this.path + "/data").tryGetChunkSizes()[0];
	
		this.type = reader.getDataSetInformation(this.path + "/data").getTypeInformation().getDataClass();
		if(this.type == HDF5DataClass.STRING) new ErrorJSON("Cannot process a matrix of String: " + this.path);
		
		this.indicesBlockSize = this.reader.getDataSetInformation(this.path + "/indices").tryGetChunkSizes()[0];
		this.nbGenes = nbGenes;
		this.nbCells = nbCells;
		
		// Read the column indexes 'indptr', required for recreating the dense matrix
		this.indptr = readLong(this.path + "/indptr"); // This one should have reasonable size (nb cells), so we load it totally
		
		// How many blocks to process?
		this.nbTotalBlocks = (int)Math.ceil((double)this.nbCells / blockSizeX);
	}
	
	private LongArray64 readLong(String path)
	{
		if(this.reader == null) new ErrorJSON("Please open the HDF5 file first");
		long length = this.reader.getDataSetInformation(path).getDimensions()[0];
		LongArray64 res = new LongArray64(length); // Know the actual size of the array, and create it sufficiently big
		int nbChunks = (int)(length / LongArray64.chunkSize()) + 1;
		for(int i = 0; i < nbChunks; i++) res.set(i, this.reader.int64().readArrayBlock(path, LongArray64.chunkSize(), i));
		return res;
	}
	
	public float[][] readSubMatrix(long start, long end, ParsingJSON json, boolean doMeta) // From cols start to end && all rows
	{
		// Create the submatrix to return
		float[][] submatrix = new float[(int)this.nbGenes][blockSizeX];
			
		// Fill the dense submatrix with values != 0
		for(long j = start; j < end; j++) // j varies accross cells ( cols )
		{	
			for(long x = indptr.get(j); x < indptr.get(j+1); x++) // x varies accross genes ( rows )
			{
				float value = -1;
				if(type == HDF5DataClass.FLOAT) value = getDataF(x);
				else if(type == HDF5DataClass.INTEGER) value = getData(x);
				int i = getIndices(x); // Original index
				
				// Put the value in the matrix
				submatrix[i][(int)(j - start)] = value; // i - start should always be in int range (because chunk size is int)
				
				if(doMeta)
				{
					// Process with annotations
					json.data.is_count_table = json.data.is_count_table && Utils.isInteger(value);
					
					// Handle biotype count per cell
					String biotype = json.data.biotypes.get(i);
					if(EnsemblDB.isProteinCoding(biotype)) json.data.proteinCodingContent.set(j, json.data.proteinCodingContent.get(j) + value);
					else if(EnsemblDB.isRibo(biotype)) json.data.ribosomalContent.set(j, json.data.ribosomalContent.get(j) + value);
					if(EnsemblDB.isMito(json.data.chromosomes.get(i))) json.data.mitochondrialContent.set(j, json.data.mitochondrialContent.get(j) + value);
	    				
	    			// Generate the sums
					json.data.depth.set(j, json.data.depth.get(j) + value);
					json.data.sum.set(i, json.data.sum.get(i) + value);
				}
			}
		}
		
		return submatrix;
	}
	
	private void readNextDataBlock()
	{
		currentDataBlockRead++;
		if(dataBlocks == null) dataBlocks = new Block(0, dataBlockSize - 1);
		else dataBlocks = new Block(dataBlocks.end + 1, dataBlocks.end + dataBlockSize);
		if(type == HDF5DataClass.FLOAT) dataBlocks.valuesF = reader.float32().readArrayBlock(this.path + "/data", dataBlockSize, currentDataBlockRead);
		else if(type == HDF5DataClass.INTEGER) dataBlocks.values = reader.int32().readArrayBlock(this.path + "/data", dataBlockSize, currentDataBlockRead);
	}
	
	private int getData(long index)
	{
		if(dataBlocks == null || dataBlocks.end < index) readNextDataBlock(); // If this index is not yet available, read an additional block
		return dataBlocks.values[(int)(index - dataBlocks.start)];
	}
	
	private float getDataF(long index)
	{
		if(dataBlocks == null || dataBlocks.end < index) readNextDataBlock(); // If this index is not yet available, read an additional block
		return dataBlocks.valuesF[(int)(index - dataBlocks.start)];
	}
	
	private void readNextIndicesBlock()
	{
		currentIndicesBlockRead++;
		if(indicesBlocks == null) indicesBlocks = new Block(0, indicesBlockSize - 1);
		else indicesBlocks = new Block(indicesBlocks.end + 1, indicesBlocks.end + indicesBlockSize);
		indicesBlocks.values = reader.int32().readArrayBlock(this.path + "/indices", indicesBlockSize, currentIndicesBlockRead);
	}
	
	private int getIndices(long index)
	{
		if(indicesBlocks == null || indicesBlocks.end < index) readNextIndicesBlock(); // If this index is not yet available, read an additional block
		return indicesBlocks.values[(int)(index - indicesBlocks.start)];
	}
}

class Block
{
	public int[] values;
	public float[] valuesF;
	public long start;
	public long end;
	
	public Block(long start, long end) 
	{
		this.start = start;
		this.end = end;
	}
}