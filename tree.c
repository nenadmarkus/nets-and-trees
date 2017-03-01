#include <stdio.h>
#include <malloc.h>
#include <memory.h>
#include <math.h>

#include <stdint.h>

/*
	
*/

#define NRANDS 16

/*
	portable time function
*/

#ifdef __GNUC__
#include <time.h>
float getticks()
{
	struct timespec ts;

	if(clock_gettime(CLOCK_MONOTONIC, &ts) < 0)
	{
		printf("clock_gettime error\n");

		return -1.0f;
	}

	return ts.tv_sec + 1e-9f*ts.tv_nsec;
}
#else
#include <windows.h>
float getticks()
{
	static double freq = -1.0;
	LARGE_INTEGER lint;

	if(freq < 0.0)
	{
		if(!QueryPerformanceFrequency(&lint))
			return -1.0f;

		freq = lint.QuadPart;
	}

	if(!QueryPerformanceCounter(&lint))
		return -1.0f;

	return (float)( lint.QuadPart/freq );
}
#endif

/*
	multiply with carry PRNG
*/

uint32_t mwcrand_r(uint64_t* state)
{
	uint32_t* m;

	//
	m = (uint32_t*)state;

	// bad state?
	if(m[0] == 0)
		m[0] = time(0);

	if(m[1] == 0)
		m[1] = time(0);

	// mutate state
	m[0] = 36969 * (m[0] & 65535) + (m[0] >> 16);
	m[1] = 18000 * (m[1] & 65535) + (m[1] >> 16);

	// output
	return (m[0] << 16) + m[1];
}

uint64_t prngglobal = 0x12345678000fffffLL;

void smwcrand(uint32_t seed)
{
	prngglobal = 0x12345678000fffffLL*seed;
}

uint32_t mwcrand()
{
	return mwcrand_r(&prngglobal);
}

/*
	
*/

#define BINTEST(r1, c1, r2, c2, pixels, ldim) ( (pixels)[(r1)*(ldim)+(c1)] > (pixels)[(r2)*(ldim)+(c2)] )

float compute_entropy(int atoms[], int inds[], int ninds)
{
	int i;
	int counts[256];
	double h;
	
	//
	memset(counts, 0, 256*sizeof(int));
	
	//
	for(i=0; i<ninds; ++i)
		++counts[ atoms[inds[i]] ];
	
	//
	h = 0.0;
	
	for(i=0; i<256; ++i)
	{
		double p;

		if(counts[i])
		{
			p = counts[i]/(double)ninds;
			
			h += -p*log2(p);
		}
	}
	
	//
	return (float)h;
}

float compute_split_entropy(int r1, int c1, int r2, int c2, uint8_t* ppixels[], int ldim, int atoms[], int inds[], int ninds)
{
	int i;
	int n0, n1;
	int counts0[256], counts1[256];
	
	double h0, h1;

	//
	n0 = 0;
	n1 = 0;
	
	memset(counts0, 0, 256*sizeof(int));
	memset(counts1, 0, 256*sizeof(int));
	
	//
	for(i=0; i<ninds; ++i)
		if( 1==BINTEST(r1, c1, r2, c2, ppixels[inds[i]], ldim) )
		{
			++n1;
			++counts1[ atoms[inds[i]] ];
		}
		else
		{
			++n0;
			++counts0[ atoms[inds[i]] ];
		}

	//
	h0 = 0.0;
	h1 = 0.0;

	for(i=0; i<256; ++i)
	{
		double p;

		if(counts0[i])
		{
			p = counts0[i]/(double)n0;
			
			h0 += -p*log2(p);
		}
		
		if(counts1[i])
		{
			p = counts1[i]/(double)n1;
			
			h1 += -p*log2(p);
		}
	}

	//
	return (float)( (n0*h0+n1*h1)/(n0+n1) );
}

int split(int r1, int c1, int r2, int c2, uint8_t* ppixels[], int ldim, int inds[], int ninds)
{
	int stop;
	int i, j;

	int n0;

	//
	stop = 0;

	i = 0;
	j = ninds - 1;

	while(!stop)
	{
		//
		while( 0==BINTEST(r1, c1, r2, c2, ppixels[inds[i]], ldim) )
		{
			if( i==j )
				break;
			else
				++i;
		}

		while( 1==BINTEST(r1, c1, r2, c2, ppixels[inds[j]], ldim) )
		{
			if( i==j )
				break;
			else
				--j;
		}

		//
		if( i==j )
			stop = 1;
		else
		{
			// swap
			inds[i] = inds[i] ^ inds[j];
			inds[j] = inds[i] ^ inds[j];
			inds[i] = inds[i] ^ inds[j];
		}
	}

	//
	n0 = 0;

	for(i=0; i<ninds; ++i)
		if( 0==BINTEST(r1, c1, r2, c2, ppixels[inds[i]], ldim) )
			++n0;

	//
	return n0;
}

int learn_subtree(int32_t* nodes, int nodeidx, int depth, int maxdepth, int atoms[], uint8_t* ppixels[], int nrows, int ncols, int inds[], int ninds)
{
	int i;
	uint8_t* n;

	int nrands;
	int rs1[NRANDS], cs1[NRANDS], rs2[NRANDS], cs2[NRANDS];

	float hs[NRANDS], hmin;
	int best;

	//
	n = (uint8_t*)&nodes[nodeidx];

	//
	if(depth == maxdepth)
	{
		int max, atom;

		#define MAXNATOMS (2048)
		int counts[MAXNATOMS];

		//
		memset(counts, 0, MAXNATOMS*sizeof(int));

		for(i=0; i<ninds; ++i)
			++counts[ atoms[inds[i]] ];

		//
		max = counts[0];
		atom = 0;

		for(i=1; i<MAXNATOMS; ++i)
			if(counts[i] > max)
			{
				max = counts[i];
				atom = i;
			}

		// construct a terminal node (leaf)
		n[0] = 0;
		n[1] = atom%256;
		n[2] = atom/256;
		n[3] = 0; // irrelevant

		//printf("%d %f\n", nodeidx-(1<<maxdepth), compute_entropy(atoms, inds, ninds));

		//
		return 1;
	}
	else if(0==ninds || 0.0f==compute_entropy(atoms, inds, ninds))
	{
		//
		n[0] = mwcrand()%nrows;
		n[1] = mwcrand()%ncols;
		n[2] = mwcrand()%nrows;
		n[3] = mwcrand()%ncols;

		//
		learn_subtree(nodes, 2*nodeidx+1, depth+1, maxdepth, atoms, ppixels, nrows, ncols, &inds[0], ninds);
		learn_subtree(nodes, 2*nodeidx+2, depth+1, maxdepth, atoms, ppixels, nrows, ncols, &inds[0], ninds);

		//
		return 1;
	}

	//
	nrands = NRANDS;

	for(i=0; i<nrands; ++i)
	{
		rs1[i] = mwcrand()%nrows;
		cs1[i] = mwcrand()%ncols;
		rs2[i] = mwcrand()%nrows;
		cs2[i] = mwcrand()%ncols;
	}

	//
	#pragma omp parallel for
	for(i=0; i<nrands; ++i)
		hs[i] = compute_split_entropy(rs1[i], cs1[i], rs2[i], cs2[i], ppixels, ncols, atoms, inds, ninds);

	//
	hmin = hs[0];
	best = 0;

	for(i=1; i<nrands; ++i)
		if(hs[i] < hmin)
		{
			hmin = hs[i];
			best = i;
		}

	// construct a nonterminal node
	n[0] = rs1[best];
	n[1] = cs1[best];
	n[2] = rs2[best];
	n[3] = cs2[best];

	// recursively buils two subtrees
	i = split(rs1[best], cs1[best], rs2[best], cs2[best], ppixels, ncols, inds, ninds);

	learn_subtree(nodes, 2*nodeidx+1, depth+1, maxdepth, atoms, ppixels, nrows, ncols, &inds[0], i);
	learn_subtree(nodes, 2*nodeidx+2, depth+1, maxdepth, atoms, ppixels, nrows, ncols, &inds[i], ninds-i);

	//
	return 1;
}

int* learn_tree(int atoms[], uint8_t* ppixels[], int nsamples, int nrows, int ncols, int tdepth)
{
	int i, nnodes;

	int32_t* tree = 0;
	int* inds = 0;

	//
	nnodes = (1<<(tdepth+1)) - 1;

	//
	tree = (int*)malloc((nnodes+1)*sizeof(int32_t));

	if(!tree)
		return 0;

	// initialize
	tree[0] = tdepth;

	memset(&tree[1], 0, nnodes*sizeof(int32_t)); // all nodes are terminal, for now

	//
	inds = (int*)malloc( nsamples*sizeof(int) );
	
	if(!inds)
	{
		//
		free(tree);

		//
		return 0;
	}
	
	for(i=0; i<nsamples; ++i)
		inds[i] = i;
	
	//
	if(!learn_subtree(&tree[1], 0, 0, tdepth, atoms, ppixels, nrows, ncols, inds, nsamples))
	{
		//
		free(tree);
		free(inds);

		//
		return 0;
	}

	//
	free(inds);

	//
	return tree;
}

/*
	
*/

int run_tree(int* tree, uint8_t pixels[], int nrows, int ncols)
{
	int i, idx, tdepth;

	//
	tdepth = tree[0];
	idx = 1;

	for(i=0; i<tdepth; ++i)
	{
		uint8_t* n = (uint8_t*)&tree[idx];
		idx = 2*idx + BINTEST(n[0], n[1], n[2], n[3], pixels, ncols);
		
	}

	return idx-(1<<tdepth);
}

/*
	
*/

#define MAXNSAMPLES 1000000

static int nsamples=0;

static int tdepth, nrows, ncols;

static uint8_t* ppixels[MAXNSAMPLES];
static int atoms[MAXNSAMPLES];

/*
	
*/

#include <lua.h>
#include <lauxlib.h>
#include <TH/TH.h>
#include <luaT.h>

int learn_tree_entry(lua_State * L)
{
	//
	THByteTensor* pixels = (THByteTensor*)luaT_checkudata(L, 1, "torch.ByteTensor");
	THByteTensor* labels = (THByteTensor*)luaT_checkudata(L, 2, "torch.ByteTensor");
	tdepth = (int)lua_tointeger(L, 3);

	//
	nsamples = pixels->size[0];
	nrows = pixels->size[1];
	ncols = pixels->size[2];

	//printf("%d %d %d\n", nsamples, nrows, ncols);

	int i;
	for(i=0; i<nsamples; ++i)
	{
		ppixels[i] = &pixels->storage->data[pixels->storageOffset + i*nrows*ncols];
		atoms[i] = labels->storage->data[labels->storageOffset + i];
	}

	//
	long int ptr = (long int)learn_tree(atoms, ppixels, nsamples, nrows, ncols, tdepth);

	//printf("%ld\n", ptr);
	lua_pushinteger(L, ptr);

	//
	return 1;
}

int run_tree_entry(lua_State * L)
{
	//
	THByteTensor* pixels = (THByteTensor*)luaT_checkudata(L, 1, "torch.ByteTensor");
	long int ptr = lua_tointeger(L, 2);

	//printf("%ld\n", ptr);

	//
	long int idx = run_tree((int*)ptr, &pixels->storage->data[pixels->storageOffset], pixels->size[0], pixels->size[1]);
	lua_pushinteger(L, idx);

	//
	return 1;
}

/*
	
*/

luaL_Reg funcs[] =
{
    {"lrn", learn_tree_entry},
    {"run", run_tree_entry},
    {0, 0}
};

int luaopen_tree(lua_State * L)
{
    luaL_register(L, "tree", funcs);
    return 1;
}
