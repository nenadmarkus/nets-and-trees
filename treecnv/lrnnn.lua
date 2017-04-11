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

require '../dataset-mnist'
mnist.download()

trainData = torch.load('mnist.t7/train_32x32.t7', 'ascii')
trainData.data = trainData.data:view(60000, 32, 32):float()

testData = torch.load('mnist.t7/test_32x32.t7', 'ascii')
testData.data = testData.data:view(10000, 32, 32):float()

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- define model to train
model = nn.Sequential()
require 'TreeCnv.lua'
model:add( nn.Bottle(nn.TreeCnv(16, 5, 5, 1, 1), 3) )
model:add( nn.View(-1, 16, 2) )
B = 32
model:add( nn.SparseLinear(16*math.pow(2, 5), B) )
model:add( nn.View(-1, 28, 28, B) )
model:add( nn.Transpose({3, 4}, {2, 3}, {3, 4}) )
model:add( nn.ReLU() )
model:add( nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1) )
model:add( nn.ReLU() )
model:add( nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1) )
model:add( nn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1) )
model:add( nn.ReLU() )
model:add( nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1) )
model:add( nn.SpatialConvolution(64, 128, 5, 5, 2, 2, 0, 0) )
model:add( nn.ReLU() )
model:add( nn.Reshape(512) )
model:add( nn.Linear(512, #classes) )

--x = torch.randn(128, 32, 32)
--y = model:forward( x )
--print(y:size())
--model:backward(x, y)
--do return end

-- retrieve parameters and gradients
parameters, gradParameters = model:getParameters()

-- verbose
print('<mnist> using model:')
print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion(nil, false)

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
   for t = 1, dataset.data:size(1), batchSize do
      -- create mini batch
      local inputs = torch.Tensor(batchSize, 32, 32)
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
         
         for i=1, batchSize do
         	--
         	confusion:add(outputs[i], targets[i])
         end

         -- return f and df/dX
         return f, gradParameters
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
      local inputs = torch.Tensor(batchSize, 32, 32)
      local targets = torch.Tensor(batchSize)
      local k = 1
      for i = t, math.min(t+batchSize-1, dataset.data:size(1)) do
         --
         inputs[k] = dataset.data[i]
         targets[k] = dataset.labels[i]
         --
         k = k + 1
      end

      -- test samples
      local outputs = model:forward(inputs)
      for i = 1, batchSize do
         --
         confusion:add(outputs[i], targets[i])
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
