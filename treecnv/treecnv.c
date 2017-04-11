#include <stdio.h>

/*
	
*/

#define BINTEST(r1, c1, r2, c2, pixels, ldim) ( (pixels)[(r1)*(ldim)+(c1)] < (pixels)[(r2)*(ldim)+(c2)] )

void run_tree(int r, int c, int treeind, short* inds, int nrows, int ncols, int ntrees, unsigned char* trees, int tdepth, float* img, int ldim, int rstride, int cstride)
{
	//
	unsigned char* tree = &trees[treeind*((1<<tdepth)-1)*4];

	img = &img[r*rstride*ldim + c*cstride];

	//
	int idx = 1;
	for(int i=0; i<tdepth; ++i)
	{
		unsigned char* n = &tree[4*(idx-1)];
		idx = 2*idx + BINTEST(n[0], n[1], n[2], n[3], img, ldim);
	}

	inds[r*ncols*ntrees + c*ntrees + treeind] = idx-(1<<tdepth);
}

void treecnv_getinds(short* inds, int nrows, int ncols, int ntrees, unsigned char* trees, int tdepth, float* img, int ldim, int rstride, int cstride)
{
	int r;
	#pragma omp parallel for
	for(r=0; r<nrows; ++r)
	{
		int c;
		for(c=0; c<ncols; ++c)
		{
			int i;
			for(i=0; i<ntrees; ++i)
				run_tree(r, c, i, inds, nrows, ncols, ntrees, trees, tdepth, img, ldim, rstride, cstride);
		}
	}
}

/*
	
*/

#include <TH/TH.h>
#include <luaT.h>

unsigned int mylog2(unsigned int val)
{
	if (val == 0) return UINT_MAX;
	if (val == 1) return 0;
	unsigned int ret = 0;
	while (val > 1)
	{
		val >>= 1;
		ret++;
	}
	return ret;
}

int getinds_entry(lua_State * L)
{
	//
	THShortTensor* inds = (THShortTensor*)luaT_checkudata(L, 1, "torch.ShortTensor");
	THFloatTensor* pixels = (THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
	THByteTensor* trees = (THByteTensor*)luaT_checkudata(L, 3, "torch.ByteTensor");
	int rstride = (int)lua_tointeger(L, 4);
	int cstride = (int)lua_tointeger(L, 5);

	int tdepth = 1 + mylog2( THByteTensor_size(trees, 1) );

	//
	treecnv_getinds(
		THShortTensor_data(inds),
		THShortTensor_size(inds, 0),
		THShortTensor_size(inds, 1),
		THShortTensor_size(inds, 2),
		THByteTensor_data(trees),
		tdepth,
		THFloatTensor_data(pixels),
		THFloatTensor_stride(pixels, 0),
		rstride,
		cstride
	);

	//
	lua_pushinteger(L, 1);
	return 1;
}

luaL_Reg funcs[] =
{
	{"getinds", getinds_entry},
	{0, 0}
};

int luaopen_treecnv(lua_State* L)
{
	luaL_register(L, "treecnv", funcs);
	return 1;
}