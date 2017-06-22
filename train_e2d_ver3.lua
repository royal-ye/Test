-- code derived from https://github.com/soumith/dcgan.torch
--

-- This version, I try to make the initial version of end-to-end training.
-- Output: 2
-- Input: img and random mask(or fixed mask), mention, If using fixed mask, input can only be img.
-- I will write the random mask version.

-- InnerSwap: For the encoder-decoder part, add a special innerswap(and return self.kbar)
--            For the Unet part, get the self.kbar derived from encoder-decoder part, 
--              using the `innerSwap_stage2` is fine.


-- Using 2 same D to tell a image.   MSE+L1, lambda = 1
-- Using squeezed_kbar!

-- Confusion:
-- 1. outputs are two images each from encoder-decoder and unet.
--    Display all these images out.

-- Extra(low priority):
-- Nocenters acc for multi-swap
-- Using maskModel instead of conv-conv to make it more compact.
-- Add param to control them.

-- ********************************************************************************************
-- It is experimental, so I using the lib_end2end, so that it don't make much noise to the others.

-- No Dropout*************************

-- ************ Version Feature **********************
-- e2d-multi
-- It is quite special, for it uses suffix of 'squeeze' scirpts, there are some differences.

require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util/util.lua')
require 'image'
require 'models'
require 'cudnn'
require 'cunn'


require 'lib_end2end/MaxCoord_squeeze'
require 'lib_end2end/InstanceNormalization'
require 'lib_end2end/innerSwap_unetPart_squeeze'   -- for the unet part
require 'lib_end2end/guideLayer_squeeze'    -- for the encoder-decoder part to provide with guidance swapping info.

require 'lib_end2end/myUtils'

opt = {
   DATA_ROOT = './datasets',         -- path to images (should have subfolders 'train', 'val', etc)
   batchSize = 1,          -- # images in batch
   loadSize = 286,         -- scale images to this size
   fineSize = 256,         --  then crop to this size
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   input_nc = 3,           -- #  of input image channels
   output_nc = 3,          -- #  of output image channels
   niter = 200,            -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   flip = 1,               -- if flip the images for data argumentation
   display = 1,            -- display samples while training. 0 = false
   display_id = 140,        -- display window id.
   display_plot = 'errL1, errG, errD, errD2',    -- which loss values to plot over time. Accepted values include a comma seperated list of: errL1, errG, and errD
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'end2end_multi_6_16',              -- name of the experiment, should generally be passed on the command line
   which_direction = 'AtoB',    -- AtoB or BtoA
   phase = 'images',             -- train, val, test, etc
   preprocess = 'regular',      -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
   nThreads = 2,                -- # threads for loading data
   save_epoch_freq = 5,        -- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
   save_latest_freq = 5000,     -- save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
   print_freq = 50,             -- print the debug information every print_freq iterations
   display_freq = 1000,          -- display the current results every display_freq iterations
   save_display_freq = 10000,    -- save the current display of results every save_display_freq_iterations
   continue_train=0,            -- if continue training, load the latest model: 1: true, 0: false
   serial_batches = 0,          -- if 1, takes images in order to make batches, otherwise takes them randomly
   serial_batch_iter = 1,       -- iter into serial image list
   checkpoints_dir = './checkpoints', -- models are saved here
   cudnn = 1,                         -- set to 0 to not use cudnn
   condition_GAN = 0,                 -- set to 0 to use unconditional discriminator
   use_GAN = 1,                       -- set to 0 to turn off GAN term
   use_L1 = 1,                        -- set to 0 to turn off L1 term
   which_model_netD = 'basic', -- selects model to use for netD
   which_model_netG = 'unet_swap_mask_end2end',  -- selects model to use for netG
   n_layers_D = 0,             -- only used if which_model_netD=='n_layers'
   lambda = 1,               -- weight on L1 term in objective
   threshold = 6/16,             -- making binary mask
   threshold2 = 6/16,            -- 128*128
   arbitray = true,             -- whether mask is arbitray
   theta = 0.5,                   -- beta is the weight on Unet loss.
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local input_nc = opt.input_nc
local output_nc = opt.output_nc
-- translation direction
local idx_A = nil
local idx_B = nil

if opt.which_direction=='AtoB' then
    idx_A = {1, input_nc}
    idx_B = {input_nc+1, input_nc+output_nc}
elseif opt.which_direction=='BtoA' then
    idx_A = {input_nc+1, input_nc+output_nc}
    idx_B = {1, input_nc}
else
    error(string.format('bad direction %s',opt.which_direction))
end

if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local data_loader = paths.dofile('data/data.lua')
print('#threads...' .. opt.nThreads)
local data = data_loader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())
tmp_d, tmp_paths = data:getBatch()

----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end


local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0


function defineG_unet_swap_mask_end2end(input_nc,output_nc,ngf, idx, conv_layers, guideLayer64, swapLayer64,guideLayer128, swapLayer128)
    -- create maskModel node
    -- use default conv_layers = 2 to create, need improvement!

    local maskModel64 = nil      -- assign correct weight after apply_weight
    local conv1 = nn.SpatialConvolution(1, 1, 4, 4, 2, 2, 1, 1)
    -- conv1.weight:fill(1/16)
    -- conv1.bias:fill(0)
    local conv2 = nn.SpatialConvolution(1, 1, 4, 4, 2, 2, 1, 1)
    -- conv2.bias:fill(0)
    -- conv2.weight:fill(1/16)
    local e1_m = - conv1
    local e2_m = e1_m - conv2
    maskModel64 = e2_m
    local maskModel128 = e1_m


  local net_G=nil
    local idx = idx or 0
  

    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 128 x 128
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 4) 
    -- input is (ngf * 4) x 32 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- nn.InstanceNormalization(ngf * 8)  

-- encoder-decoder part
    local d1_d = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8) -- nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d2_d = d1_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8) -- nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d3_d = d2_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8) -- nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d4_d = d3_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local d5_d = d4_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local d6_d = d5_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 2) 
    -- input is (ngf * 2) x 64 x 64
    local d6_d_gd = {d6_d, maskModel64} - nn.JoinTable(2) - nn.ReLU(true) - guideLayer64  -- guide layer fork  , as maskModel is positive, so it is fine.
    local d7_d = d6_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf)
    -- input is (ngf) x128 x 128
    local d7_d_gd = {d7_d, maskModel128} - nn.JoinTable(2) - nn.ReLU(true) - guideLayer128
    local d8_d = d7_d - nn.ReLU(true) - nn.SpatialFullConvolution(ngf, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 256 x 256
    
    local o1_d = d8_d - nn.Tanh() 

-- unet part
    -- input is (ngf * 8) x 1 x 1
    local d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8) -- nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d1 = {d1_,e7} - nn.JoinTable(2)
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8) -- nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8) -- nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true)  - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true)  - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local d6 = {d6_,e2} - nn.JoinTable(2) - nn.ReLU(true) 
    local d6_tmp = {d6, d6_d_gd} - nn.JoinTable(2)   -- add guidance info, should behind the relu of data join.
    local d7_ = d6_tmp - swapLayer64 - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf)
    -- input is (ngf) x128 x 128
    local d7 = {d7_,e1} - nn.JoinTable(2) - nn.ReLU(true)
    local d7_tmp = {d7, d7_d_gd} - nn.JoinTable(2)
    local d8 = d7_tmp - swapLayer128 - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 256 x 256
    local o1 = d8 - nn.Tanh()  

    netG = nn.gModule({e1, e1_m},{o1_d, o1})

    print('netG...')
    printNet(netG)
    graph.dot(netG.fg,'MLP','forward')
    graph.dot(netG.bg,'MLP','backward')
    return netG
end

function defineG(input_nc, output_nc, ngf)
    -- Add maskModel and swapLayer
    -- As maskModel must be a nnNode, so cannot create a maskModel(nn.Module), create it in
    -- defineG_unet_swap_mask, can we do something to make maskModel(nn.Module) possible?
    -- That will be a nicer solve.
    local conv_layers = 2
    local mask_thred = 1
    local swap_sz = 1
    local stride = 1
    -- local maskModel = nn.MaskModel(conv_layers, opt.threshold)
    -- print(maskModel)

    -- the name should be innerSwap_64, innerSwap_128
    local swapLayer64 = nn.InnerSwap_unetPart('innerSwap_64',swap_sz, opt.threshold, mask_thred)
    local guideLayer64 = nn.GuideLayer(swap_sz, stride, mask_thred)

    local swapLayer128 = nn.InnerSwap_unetPart('innerSwap_128',swap_sz, opt.threshold2, mask_thred)
    local guideLayer128 = nn.GuideLayer(swap_sz, stride, mask_thred)

    local netG = nil
    if     opt.which_model_netG == "encoder_decoder" then netG = defineG_encoder_decoder(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet" then netG = defineG_unet(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet_swap_mask_end2end" then netG = defineG_unet_swap_mask_end2end(input_nc, output_nc, ngf, 1, conv_layers, guideLayer64, swapLayer64, guideLayer128, swapLayer128)
    elseif opt.which_model_netG == "unet_128" then netG = defineG_unet_128(input_nc, output_nc, ngf)
    else error("unsupported netG model")
    end
   
    netG:apply(weights_init)

--********** reassgin the correct maskModel weigth**************
printNet(netG)
	netG.modules[71].weight:fill(1/16)
	netG.modules[72].weight:fill(1/16)
    return netG
end

function defineD(input_nc, output_nc, ndf)
    local netD = nil
    local netD2 = nil
    if opt.condition_GAN==1 then
        input_nc_tmp = input_nc
    else
        input_nc_tmp = 0 -- only penalizes structure in output channels 
    end
    
    if     opt.which_model_netD == "basic" then netD = defineD_basic(input_nc_tmp, output_nc, ndf); netD2 = defineD_basic(input_nc_tmp, output_nc, ndf)
    elseif opt.which_model_netD == "n_layers" then netD = defineD_n_layers(input_nc_tmp, output_nc, ndf, opt.n_layers_D); 
                                                    netD2 = defineD_n_layers(input_nc_tmp, output_nc, ndf, opt.n_layers_D); 
    else error("unsupported netD model")
    end
    
    netD:apply(weights_init)
    netD2:apply(weights_init)
    
    return netD, netD2
end

local mask_global = torch.ByteTensor(opt.batchSize, opt.fineSize, opt.fineSize)
local res = 0.05 -- the lower it is, the more continuous the output will be. 0.01 is too small and 0.1 is too large
local density = 0.25
local MAX_SIZE = 10000
local low_pattern = torch.Tensor(res*MAX_SIZE, res*MAX_SIZE):uniform(0,1):mul(255)
local pattern = image.scale(low_pattern, MAX_SIZE, MAX_SIZE,'bicubic')
low_pattern = nil
collectgarbage()
pattern:div(255)
pattern = torch.lt(pattern,density):byte()  -- 25% 1s and 75% 0s
pattern = pattern:byte()
print('...Random pattern generated')



-- load saved models and finetune
if opt.continue_train == 1 then
   print('loading previously trained netG...')
   netG = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), opt)
   print('loading previously trained netD...')
   netD = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), opt)
   netD2 = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D2.t7'), opt)
else
  print('define model netG...')
  netG = defineG(input_nc, output_nc, ngf)
  print('define model netD...')
  netD, netD2 = defineD(input_nc, output_nc, ndf)
end

print('netD...')
print(netD)


local criterion = nn.MSECriterion()
local criterionAE = nn.AbsCriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD2 = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
local real_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_B_encDec = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local real_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local fake_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local fake_AB_encDec = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local errD, errD2, errG, errL1 = 0, 0, 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------

if opt.gpu > 0 then
   print('transferring to gpu...')
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   real_A = real_A:cuda();
   real_B = real_B:cuda(); fake_B = fake_B:cuda(); fake_B_encDec = fake_B_encDec:cuda();
   real_AB = real_AB:cuda(); fake_AB = fake_AB:cuda(); fake_AB_encDec = fake_AB_encDec:cuda();
   if opt.cudnn==1 and opt.continue_train == 0 then
      netG = util.cudnn(netG); netD = util.cudnn(netD); netD2 = util.cudnn(netD2)
   end
   netD:cuda(); netD2:cuda(); netG:cuda(); criterion:cuda(); criterionAE:cuda();
   print('done')
else
	print('running model on CPU')
end


local parametersD, gradParametersD = netD:getParameters()
local parametersD2, gradParametersD2 = netD2:getParameters()
local parametersG, gradParametersG = netG:getParameters()



if opt.display then disp = require 'display' end


if opt.arbitray == false then
    mask_global = image.load('mask_fix.png',1,'byte')
    mask_global = torch.div(mask_global,255):byte()
end

data_path = {}

function createRealFake()
    if opt.arbitray == true then
        local mask_global_tmp = torch.ByteTensor(opt.batchSize, opt.fineSize, opt.fineSize)
        mask_global = create_gMask(pattern, mask_global_tmp, MAX_SIZE, opt)
    end

    -- load real
    data_tm:reset(); data_tm:resume()
    local real_data
	real_data, data_path = data:getBatch()
	print(data_path)
	print()
    real_data = real_data:cuda()  -- This should be omitted.
    data_tm:stop()

    
-- mask_global must be byteTensor    
    real_A:copy(real_data[{ {}, idx_A, {}, {} }])
    real_A[{{},{1},{},{}}][mask_global] = 2*117.0/255.0 - 1.0
    real_A[{{},{2},{},{}}][mask_global] = 2*104.0/255.0 - 1.0
    real_A[{{},{3},{},{}}][mask_global] = 2*123.0/255.0 - 1.0

    real_B:copy(real_data[{ {}, idx_B, {}, {} }])

    mask_global = mask_global:view(1,mask_global:size(1),mask_global:size(2),mask_global:size(3))

    -- real_A is input(we use GAN preliminary filled images), and real_B is groundTruth
    if opt.condition_GAN==1 then
        real_AB = torch.cat(real_A,real_B,2)
    else
        real_AB = real_B -- unconditional GAN, only penalizes structure in B
    end

    -- transfer mask to cudaTensor()
    mask_global = mask_global:cuda()

    fake_tb = netG:forward({real_A, mask_global})
    fake_B_encDec = fake_tb[1]
    fake_B = fake_tb[2]
  
    if opt.condition_GAN==1 then
        fake_AB = torch.cat(real_A,fake_B,2)
        fake_AB_encDec = torch.cat(real_A, fake_B_encDec, 2)
    else
        fake_AB = fake_B -- unconditional GAN, only penalizes structure in B
        fake_AB_encDec = fake_B_encDec
    end
    
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersD:zero()
    
    -- Real
    -- train netD with (real, real_label)
    local output = netD:forward(real_AB)
    local label = torch.FloatTensor(output:size()):fill(real_label)
    if opt.gpu>0 then 
    	label = label:cuda()
    end
    
    local errD_real = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(real_AB, df_do)

-- Now only use the fake_AB to update D, later we should change D to make it accept two inputs?
    -- Fake
    -- train netD with (fake_AB, fake_label)
    local output = netD:forward(fake_AB)

    label:fill(fake_label)
    local errD_fake = criterion:forward(output, label)

    local df_do = criterion:backward(output, label)



    netD:backward(fake_AB, df_do)
    -- netD:backward(fake_AB, df_do)  I think we should change D to accept two inputs!***************************
    
    errD = (errD_real + errD_fake)/2
    
    return errD, gradParametersD
end

local fDx2 = function(x)
    netD2:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersD2:zero()
    
    -- Real
    -- train netD with (real, real_label)
    local output = netD2:forward(real_AB)
    local label = torch.FloatTensor(output:size()):fill(real_label)
    if opt.gpu>0 then 
    	label = label:cuda()
    end
    
    local errD_real = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD2:backward(real_AB, df_do)

    -- Fake, using fake_AB_encDec
    local output = netD2:forward(fake_AB_encDec)

    label:fill(fake_label)
    local errD_fake = criterion:forward(output, label)

    local df_do = criterion:backward(output, label)

    netD2:backward(fake_AB_encDec, df_do)

    errD2 = (errD_real + errD_fake)/2
    
    return errD2, gradParametersD2
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netD2:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersG:zero()
    
    -- GAN loss
    local df_dg = torch.zeros(fake_B:size())
    local df_dg_encDec = torch.zeros(fake_B_encDec:size())  -- added
    if opt.gpu>0 then 
    	df_dg = df_dg:cuda();
        df_dg_encDec = df_dg_encDec:cuda()
    end

    -- output are netD:forward(fake_AB), just a serials of labels of float number.
    -- then We need to minimize the loss between output and real_label
    if opt.use_GAN==1 then
       local output = netD.output -- netD:forward{input_A,input_B} was already executed in fDx, so save computation
       local label = torch.FloatTensor(output:size()):fill(real_label) -- fake labels are real for generator cost
       if opt.gpu>0 then 
       	label = label:cuda();
       end
       errG = criterion:forward(output, label)
       local df_do = criterion:backward(output, label)
       -- If we use cGAN, then assume that the grad is bs*6*h*w, then we only need the grad 
       -- of fake_B. So narrow(2, ....)
       df_dg = netD:updateGradInput(fake_AB, df_do):narrow(2,fake_AB:size(2)-output_nc+1, output_nc)
       df_dg_encDec = netD2:updateGradInput(fake_AB_encDec, df_do):narrow(2,fake_AB_encDec:size(2)-output_nc+1, output_nc)
    else
        errG = 0
    end
    
    -- unary loss
    local df_do_AE = torch.zeros(fake_B:size())
    local df_do_AE_encDec = torch.zeros(fake_B_encDec:size())
    if opt.gpu>0 then 
    	df_do_AE = df_do_AE:cuda();
    	df_do_AE_encDec = df_do_AE_encDec:cuda();
    end
    if opt.use_L1==1 then
       errL1 = criterionAE:forward(fake_B, real_B)
       errL1_encDec = criterionAE:forward(fake_B_encDec, real_B)

       df_do_AE = criterionAE:backward(fake_B, real_B)
       df_do_AE_encDec = criterionAE:backward(fake_B_encDec, real_B)
    else
        errL1 = 0
        errL1_encDec = 0
    end

    -- need to mention!
 --   netG:backward(real_A, df_dg+ df_do_AE:mul(opt.lambda))

    local com_encDec =  df_dg_encDec + df_do_AE_encDec:mul(opt.lambda)  
    local com_unet = df_dg + df_do_AE:mul(opt.lambda)
    netG:backward({real_A, mask_global}, { com_encDec:mul(1 - opt.theta), com_unet:mul(opt.theta) } )  -- I doubt that if I just use real_A, then the maskModel path won't calcute.

    return errG, gradParametersG
end




-- train
local best_err = nil
paths.mkdir(opt.checkpoints_dir)
paths.mkdir(opt.checkpoints_dir .. '/' .. opt.name)

-- save opt
file = torch.DiskFile(paths.concat(opt.checkpoints_dir, opt.name, 'opt.txt'), 'w')
file:writeObject(opt)
file:close()

-- parse diplay_plot string into table
opt.display_plot = string.split(string.gsub(opt.display_plot, "%s+", ""), ",")
for k, v in ipairs(opt.display_plot) do
    if not util.containsValue({"errG", "errD","errD2", "errL1"}, v) then 
        error(string.format('bad display_plot value "%s"', v)) 
    end
end

-- display plot config
local plot_config = {
  title = "Loss over time",
  labels = {"epoch", unpack(opt.display_plot)},
  ylabel = "loss",
}

-- display plot vars
local plot_data = {}
local plot_win

local counter = 0
for epoch = 1, opt.niter do
    epoch_tm:reset()
    for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
        tm:reset()
        
        -- load a batch and run G on that batch
        createRealFake()
        
        -- (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        if opt.use_GAN==1 then optim.adam(fDx, parametersD, optimStateD); optim.adam(fDx2, parametersD2, optimStateD2) end


        -- (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        optim.adam(fGx, parametersG, optimStateG)

        -- display
        counter = counter + 1
        if counter % opt.display_freq == 0 and opt.display then
            createRealFake()
            if opt.preprocess == 'colorization' then 
                local real_A_s = util.scaleBatch(real_A:float(),100,100)
                local fake_B_s = util.scaleBatch(fake_B:float(),100,100)
                local real_B_s = util.scaleBatch(real_B:float(),100,100)
                disp.image(util.deprocessL_batch(real_A_s), {win=opt.display_id, title=opt.name .. ' input'})
                disp.image(util.deprocessLAB_batch(real_A_s, fake_B_s), {win=opt.display_id+1, title=opt.name .. ' output'})
                disp.image(util.deprocessLAB_batch(real_A_s, real_B_s), {win=opt.display_id+2, title=opt.name .. ' target'})
            else
                disp.image(util.deprocess_batch(util.scaleBatch(real_A:float(),200,200)), {win=opt.display_id, title=opt.name .. ' input'})
                disp.image(util.deprocess_batch(util.scaleBatch(fake_B:float(),200,200)), {win=opt.display_id+1, title=opt.name .. ' output'})
                disp.image(util.deprocess_batch(util.scaleBatch(fake_B_encDec:float(),200,200)), {win=opt.display_id+3, title=opt.name .. ' output_encDec'})
                disp.image(util.deprocess_batch(util.scaleBatch(real_B:float(),200,200)), {win=opt.display_id+2, title=opt.name .. ' target'})
            end
        end
      
        -- write display visualization to disk
        --  runs on the first batchSize images in the opt.phase set
        if counter % opt.save_display_freq == 0 and opt.display then
            local serial_batches=opt.serial_batches
            opt.serial_batches=1
            opt.serial_batch_iter=1
            
            local image_out = nil
            local N_save_display = 10 
            local N_save_iter = torch.max(torch.Tensor({1, torch.floor(N_save_display/opt.batchSize)}))
            for i3=1, N_save_iter do
            
                createRealFake()
                print('save to the disk')
                if opt.preprocess == 'colorization' then 
                    for i2=1, fake_B:size(1) do
                        if image_out==nil then image_out = torch.cat(util.deprocessL(real_A[i2]:float()),util.deprocessLAB(real_A[i2]:float(), fake_B[i2]:float()),3)/255.0
                        else image_out = torch.cat(image_out, torch.cat(util.deprocessL(real_A[i2]:float()),util.deprocessLAB(real_A[i2]:float(), fake_B[i2]:float()),3)/255.0, 2) end
                    end
                else
                    for i2=1, fake_B:size(1) do
                        if image_out==nil then image_out = torch.cat({util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()), util.deprocess(fake_B_encDec[i2]:float())}, 3)
                        else image_out = torch.cat(image_out, torch.cat({util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()), util.deprocess(fake_B_encDec[i2]:float())},3), 2) end
                    end
                end
            end
            image.save(paths.concat(opt.checkpoints_dir,  opt.name , counter .. '_train_res.png'), image_out)
            
            opt.serial_batches=serial_batches
        end
        
        -- logging and display plot
        if counter % opt.print_freq == 0 then
            local loss = {errG=errG and errG or -1, errD=errD and errD or -1, errD2=errD2 and errD2 or -1, errL1=errL1 and errL1 or -1}
            local curItInBatch = ((i-1) / opt.batchSize)
            local totalItInBatch = math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize)
            print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                    .. '  Err_G: %.4f  Err_D: %.4f  Err_D2: %.4f  ErrL1: %.4f'):format(
                     epoch, curItInBatch, totalItInBatch,
                     tm:time().real / opt.batchSize, data_tm:time().real / opt.batchSize,
                     errG, errD, errD2, errL1))
           
            local plot_vals = { epoch + curItInBatch / totalItInBatch }
            for k, v in ipairs(opt.display_plot) do
              if loss[v] ~= nil then
               plot_vals[#plot_vals + 1] = loss[v] 
             end
            end

            -- update display plot
            if opt.display then
              table.insert(plot_data,plot_vals )--{epoch, errG,errD,errG_l2}
              plot_config.win = plot_win
              plot_win = disp.plot(plot_data, plot_config)
            end
        
        -- save latest modelfuyt5
          if counter % opt.save_latest_freq == 0 then
              print(('saving the latest model (epoch %d, iters %d)'):format(epoch, counter))
              torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), netG:clearState())
              torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), netD:clearState())
              torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D2.t7'), netD2:clearState())
          end
      end
    end
    
    
    parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
    parametersD2, gradParametersD2 = nil, nil -- nil them to avoid spiking memory
    parametersG, gradParametersG = nil, nil
    
    if epoch % opt.save_epoch_freq == 0 then
        torch.save(paths.concat(opt.checkpoints_dir, opt.name,  epoch .. '_net_G.t7'), netG:clearState())
        torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_D.t7'), netD:clearState())
        torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_D2.t7'), netD2:clearState())
    end
    
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
    parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
    parametersD2, gradParametersD2 = netD2:getParameters() -- reflatten the params and get them
    parametersG, gradParametersG = netG:getParameters()
end

