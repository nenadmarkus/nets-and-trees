--
require 'nn'

require 'treecnv'

--
--
--

TreeCnv, parent = torch.class('nn.TreeCnv', 'nn.Module')

function TreeCnv:__init(ntrees, tdepth, wsize, rstride, cstride)
	--
	self.ntrees = ntrees
	self.tdepth = tdepth
	self.wsize = wsize
	self.rstride = rstride
	self.cstride = cstride

	--
	self.bintests = torch.rand(ntrees, 2^tdepth-1, 4):mul(wsize):floor():byte()
	--
	self.indadd = torch.ShortTensor(ntrees)
	for k=1, ntrees do
		--
		self.indadd[k] = 1 + (k-1)*2^tdepth
	end
	--
	self.weight = nil
	self.gradWeight = nil
	--
	self.bias = nil
	self.gradBias = nil
end

function TreeCnv:updateOutput(input)
	--
	local b = input:size(1)
	local h = math.floor((input:size(2) - self.wsize)/self.rstride + 1)
	local w = math.floor((input:size(3) - self.wsize)/self.cstride + 1)
	--
	if self.bintests:type() ~= 'torch.ByteTensor' then
		--
		self.bintests = torch.rand(self.ntrees, 2^self.tdepth-1, 4):mul(self.wsize):floor():byte()
	end

	--
	self.inds = torch.ShortTensor(b, h, w, self.ntrees, 2)
	for i=1, b do
		--
		local inds = torch.ShortTensor(h, w, self.ntrees):zero()
		treecnv.getinds(inds, input[i], self.bintests, self.rstride, self.cstride)
		--
		inds = inds + self.indadd:view(1, 1, self.ntrees):expand(h, w, self.ntrees) 
		inds = inds:view(h, w, self.ntrees, 1):expand(h, w, self.ntrees, 2):contiguous()
		inds:indexFill(4, torch.LongTensor{2}, 1)
		--
		self.inds[i] = inds
	end

	--
	self.output = self.inds:float()
	return self.output
end

function TreeCnv:updateGradInput(input, gradOutput)
	--
	return nil
end

function TreeCnv:accGradParameters(input, gradOutput)
	--
end

function TreeCnv:__tostring__()
	--
	return 'nn.TreeCnv(' .. self.ntrees .. ', ' .. self.tdepth .. ', ' .. self.wsize .. ', ' .. self.rstride .. ', ' .. self.cstride .. ')'
end

function TreeCnv:clearState()
	--
	self.inds = nil
	--
	return parent.clearState(self)
end