--
--
require 'torch'
require 'nn'
require 'optim'
require 'paths'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -p,--plot                                plot while training
   --coefL2           (default 0)           L2 penalty on the weights
   --type             (default "nn")        predictor architecture: nn | rrr | full
   --save             (default "")          where to save the model
]]

-- fix seed
--torch.manualSeed(1)

-- use floats
torch.setdefaulttensortype('torch.FloatTensor')

--
--
batchSize = 400
learningRate = 0.001

----------------------------------------------------------------------
-- get dataset
--
trainData = torch.load('mnist.t7/TRN.t7')
testData = torch.load('mnist.t7/TST.t7')

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- define model to train
model = nn.Sequential()

if opt.type=='nn' then
   --
   local B = 4 -- bottleneck size
   model:add(nn.SparseLinear(trainData.ntrees*math.pow(2, trainData.tdepth), B))
   model:add(nn.Tanh())
   model:add(nn.Linear(B, 2*B))
   model:add(nn.Tanh())
   model:add(nn.Linear(2*B, #classes))
elseif opt.type=='rrr' then
   --
   local B = 4 -- bottleneck size
   model:add(nn.SparseLinear(trainData.ntrees*math.pow(2, trainData.tdepth), B))
   model:add(nn.Linear(B, #classes))
elseif opt.type=='full' then
   --
   model:add(nn.SparseLinear(trainData.ntrees*math.pow(2, trainData.tdepth), #classes))
end

-- retrieve parameters and gradients
parameters, gradParameters = model:getParameters()

-- verbose
print('<mnist> using model:')
print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat('logs', 'train.log'))
testLogger = optim.Logger(paths.concat('logs', 'test.log'))

-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t = 1, dataset.data:size(1),batchSize do
      -- create mini batch
      local inputs = torch.Tensor(batchSize, dataset.ntrees, 2)
      local targets = torch.Tensor(batchSize)
      local k = 1
      for i = t, math.min(t+batchSize-1, dataset.data:size(1)) do
         --
         inputs[k] = dataset.data[i]
         targets[k] = dataset.labels[i]
         --
         k = k + 1
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)

         -- update confusion
         for i = 1, batchSize do
            confusion:add(outputs[i], targets[i])
         end

         -- return f and df/dX
         return f,gradParameters
      end

      --
      optimState = optimState or {
         learningRate = learningRate,
         weightDecay = opt.coefL2
      }
      optim.rmsprop(feval, parameters, optimState)

      -- disp progress
      xlua.progress(t, dataset.data:size(1))
   end

   -- time taken
   time = sys.clock() - time
   time = time / dataset.data:size(1)
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- next epoch
   epoch = epoch + 1
end

-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1, dataset.data:size(1), batchSize do
      -- disp progress
      xlua.progress(t, dataset.data:size(1))

      -- create mini batch
      local inputs = torch.Tensor(batchSize, dataset.ntrees, 2)
      local targets = torch.Tensor(batchSize)
      local k = 1
      for i = t,math.min(t+batchSize-1,dataset.data:size(1)) do
         --
         inputs[k] = dataset.data[i]
         targets[k] = dataset.labels[i]
         --
         k = k + 1
      end

      -- test samples
      local preds = model:forward(inputs)

      -- confusion:
      for i = 1, batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset.data:size(1)
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()
end

----------------------------------------------------------------------
-- and train!
--
while true do
   -- train/test
   train(trainData)
   test(testData)
   -- plot errors
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      trainLogger:plot()
      testLogger:plot()
   end
   --
   if opt.save~="" then
       --
       --model:clearState()
       torch.save(opt.save, model)
   end
end
