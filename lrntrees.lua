--
--
--

require 'torch'

NTREES = 256
TDEPTH = 12

--
-- download and load MNIST data
--

require 'dataset-mnist'
mnist.download()

trn = torch.load('mnist.t7/train_32x32.t7', 'ascii')
trn.data = trn.data:view(60000, 32, 32)

tst = torch.load('mnist.t7/test_32x32.t7', 'ascii')
tst.data = tst.data:view(10000, 32, 32)

--require 'image'
--image.save('mn.png', tst.data[1])

--
--
--

require 'tree'
treeptrs = {}
t = sys.clock()
for i=1, NTREES do
	--
	table.insert(treeptrs, tree.lrn(trn.data, trn.labels-1, TDEPTH))
end
print('* ' .. NTREES .. ' trees learned in ' .. sys.clock()-t .. ' [s]')

--
--
--

function tree_inds_to_sparse_vectors(pixels, treeptrs, tdepth)
	-- 
	local T = torch.FloatTensor(pixels:size(1), #treeptrs, 2)
	for i=1, pixels:size(1) do
		--
		local tbl = {}
		for j=1, #treeptrs do
			--
			table.insert(tbl, 1 + tree.run(pixels[i], treeptrs[j]) + (j-1)*math.pow(2, tdepth))
			table.insert(tbl, 1)
		end
		--
		T[i] = torch.FloatTensor(tbl):view(#treeptrs, 2)
	end
	--
	return T
end

TRN = tree_inds_to_sparse_vectors(trn.data, treeptrs, TDEPTH)
TST = tree_inds_to_sparse_vectors(tst.data, treeptrs, TDEPTH)

--
--
--

torch.save('mnist.t7/TRN.t7', {data=TRN, labels=trn.labels, tdepth=TDEPTH, ntrees=NTREES})
torch.save('mnist.t7/TST.t7', {data=TST, labels=tst.labels, tdepth=TDEPTH, ntrees=NTREES})