require 'torch'
require 'optim'
require 'nn'

-- The data relates the amount of corn produced, given certain amounts
-- of fertilizer and insecticide

-- In this example, we want to be able to predict the amount of
-- corn produced, given the amount of fertilizer and intesticide used.
-- In other words: fertilizer & insecticide are our two input variables,
-- and corn is our target value.

--  {corn, fertilizer, insecticide}
data = torch.Tensor{
   {40,  6,  4},
   {44, 10,  4},
   {46, 12,  5},
   {48, 14,  7},
   {52, 16,  9},
   {58, 18, 12},
   {60, 22, 14},
   {68, 24, 20},
   {74, 26, 21},
   {80, 32, 24}
}

model = nn.Sequential()                 -- define the container
ninputs = 2; noutputs = 1
model:add(nn.Linear(ninputs, noutputs)) -- define the only module


criterion = nn.MSECriterion()



-- SGD is a good optimization algorithm when the amount of training data
-- is large, and estimating the gradient of the loss function over the
-- entire training set is too costly.

x, dl_dx = model:getParameters()

-- In the following code, we define a closure, feval, which computes
-- the value of the loss function at a given point x, and the gradient of
-- that function with respect to x. x is the vector of trainable weights,
-- which, in this example, are all the weights of the linear matrix of
-- our model, plus one bias.

feval = function(x_new)
   -- set x to x_new, if differnt
   -- (in this simple example, x_new will typically always point to x,
   -- so the copy is really useless)
   if x ~= x_new then
      x:copy(x_new)
   end

   -- select a new training sample
   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > (#data)[1] then _nidx_ = 1 end

   local sample = data[_nidx_]
   local target = sample[{ {1} }]      -- this funny looking syntax allows
   local inputs = sample[{ {2,3} }]    -- slicing of arrays.

   -- reset gradients (gradients are always accumulated, to accomodate
   -- batch methods)
   dl_dx:zero()

   -- evaluate the loss function and its derivative wrt x, for that sample
   local loss_x = criterion:forward(model:forward(inputs), target)
   model:backward(inputs, criterion:backward(model.output, target))

   -- return loss(x) and dloss/dx
   return loss_x, dl_dx
end

-- Given the function above, we can now easily train the model using SGD.
-- For that, we need to define four key parameters:
--   + a learning rate: the size of the step taken at each stochastic
--     estimate of the gradient
--   + a weight decay, to regularize the solution (L2 regularization)
--   + a momentum term, to average steps over time
--   + a learning rate decay, to let the algorithm converge more precisely

sgd_params = {
   learningRate = 1e-3,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

-- We're now good to go... all we have left to do is run over the dataset
-- for a certain number of iterations, and perform a stochastic update
-- at each iteration. The number of iterations is found empirically here,
-- but should typically be determinined using cross-validation.

-- we cycle 1e4 times over our training data
for i = 1,1e5 do

   -- this variable is used to estimate the average loss
   current_loss = 0

   -- an epoch is a full loop over our training data
   for i = 1,(#data)[1] do

      -- optim contains several optimization algorithms.
      -- All of these algorithms assume the same parameters:
      --   + a closure that computes the loss, and its gradient wrt to x,
      --     given a point x
      --   + a point x
      --   + some parameters, which are algorithm-specific

      _,fs = optim.sgd(feval,x,sgd_params)

      -- Functions in optim all return two things:
      --   + the new x, found by the optimization method (here SGD)
      --   + the value of the loss functions at all points that were used by
      --     the algorithm. SGD only estimates the function once, so
      --     that list just contains one value.

      current_loss = current_loss + fs[1]
   end

   -- report average error on epoch
   current_loss = current_loss / (#data)[1]
   print('current loss = ' .. current_loss)

end


----------------------------------------------------------------------
-- 5. Test the trained model.

-- Now that the model is trained, one can test it by evaluating it
-- on new samples.

-- The text solves the model exactly using matrix techniques and determines
-- that
--   corn = 31.98 + 0.65 * fertilizer + 1.11 * insecticides

-- We compare our approximate results with the text's results.

text = {40.32, 42.92, 45.33, 48.85, 52.37, 57, 61.82, 69.78, 72.19, 79.42}

print('id  approx   text')
for i = 1,(#data)[1] do
   local myPrediction = model:forward(data[i][{{2,3}}])
   print(string.format("%2d  %6.2f %6.2f", i, myPrediction[1], text[i]))
end
