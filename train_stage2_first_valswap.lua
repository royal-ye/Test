-- usage example: DATA_ROOT=/path/to/data/ which_direction=BtoA name=expt1 th train.lua 
--
-- code derived from https://github.com/soumith/dcgan.torch
--

-- In this version, Load the model(trained in encoder-decoder without swap)
-- Then get the inner feature, then inner-swap it, then get the self.kbar.
-- Use the kbar in innerSwap_stage2 as guidance for swap.

-- ************* using encoder-decoder to init u-net.
-- ******************************
-- *********** Enc-dec pix ---> Unet 
-- Fix the first part of Unet!

-- As reading images are random, so for the first epoch, we cannot store `latter_data` for each image.
-- By check the the name of loading image, reload it if has loaded.

-- need to math.huge a lot of settings in case of calling to ceateFake 




require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util/util.lua')
require 'image'
require 'models'
require 'cudnn'
require 'cunn'

require 'lib_end2end/spZeroGrad'
local layinit = require 'lib_end2end/layinit'
require 'lib_end2end/MaxCoord'
require 'lib_end2end/InstanceNormalization'
require 'lib_end2end/innerSwap_stage2_val_swap3or1'  -- ********** remember in this script, the Nonpar should be the same as here, maybe use different fucntion name 
                                -- to avoid mistakes by accident.
require 'lib_end2end/myUtils'

opt = {
   DATA_ROOT = './datasets/newdata',         -- path to images (should have subfolders 'train', 'val', etc)
   batchSize = 1,          -- # images in batch
   loadSize = 286,         -- scale images to this size
   fineSize = 256,         --  then crop to this size
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   input_nc = 3,           -- #  of input image channels
   output_nc = 3,          -- #  of output image channels
   niter = 1,            -- #  can only be 1!!!!!!!!
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   flip = 0,               -- if flip the images for data argumentation
   display = 1,            -- display samples while training. 0 = false
   display_id = 140,        -- display window id.
   display_plot = 'errL1, errG, errD',    -- which loss values to plot over time. Accepted values include a comma seperated list of: errL1, errG, and errD
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'fixed_stage2_64_6_16',              -- name of the experiment, should generally be passed on the command line
   which_direction = 'AtoB',    -- AtoB or BtoA
   phase = 'train_10',             -- train, val, test, etc
   preprocess = 'regular',      -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
   nThreads = 2,                -- # threads for loading data
   save_epoch_freq = math.huge,        -- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
   save_latest_freq = math.huge,     -- save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
   print_freq = 50,             -- print the debug information every print_freq iterations
   display_freq = math.huge,          -- display the current results every display_freq iterations
   save_display_freq = math.huge,    -- save the current display of results every save_display_freq_iterations
   continue_train=0,            -- if continue training, load the latest model: 1: true, 0: false
   serial_batches = 1,          -- if 1, takes images in order to make batches, otherwise takes them randomly
   serial_batch_iter = 1,       -- iter into serial image list
   checkpoints_dir = './checkpoints/3_1', -- models are saved here
   cudnn = 1,                         -- set to 0 to not use cudnn
   condition_GAN = 1,                 -- set to 0 to use unconditional discriminator
   use_GAN = 1,                       -- set to 0 to turn off GAN term
   use_L1 = 1,                        -- set to 0 to turn off L1 term
   which_model_netD = 'basic', -- selects model to use for netD
   which_model_netG = 'unet_swap',  -- selects model to use for netG
   n_layers_D = 0,             -- only used if which_model_netD=='n_layers'
   lambda = 100,               -- weight on L1 term in objective
   threshold = 6/16,             -- making binary mask
 --   arbitray = false,              -- define whether to use arbitray mask or the fix mask used in pretrained model(encoder_decoder)
    pretrained = './checkpoints/en_de_pix_instance/180_net_G.t7',
    features_dir = './saved_features/',
    type = 'data_extraction'    -- define whether train_stage2.lua works as data_extracter or training pattern.
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
local data_loader = paths.dofile('data/dataC.lua')
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


function defineG_unet_swap(input_nc,output_nc,ngf, idx)
  local net_G=nil
    local idx = idx or 0
  

    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- cannot directly add ZeroGrad to e1, beacuse the nngraph module can only accept type like
    -- `local m = - nn.Module`, I hate this restrict.
    local e1_sp = e1 - nn.ZeroGrad()  
    -- input is (ngf) x 128 x 128
    local e2 = e1_sp - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.ZeroGrad() - nn.InstanceNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.ZeroGrad() - nn.InstanceNormalization(ngf * 4) 
    -- input is (ngf * 4) x 32 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.ZeroGrad() - nn.InstanceNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.ZeroGrad() - nn.InstanceNormalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.ZeroGrad() - nn.InstanceNormalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.ZeroGrad() - nn.InstanceNormalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.ZeroGrad() -- nn.InstanceNormalization(ngf * 8) -- This should be removed for e2d
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
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local d6 = {d6_,e2} - nn.JoinTable(2)
    local d7_ = d6 - nn.ReLU(true) - swap2 - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.InstanceNormalization(ngf)
    -- input is (ngf) x128 x 128
    local d7 = {d7_,e1_sp} - nn.JoinTable(2)
    local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 256 x 256
    local o1 = d8 - nn.Tanh()     

    netG = nn.gModule({e1},{o1})
    return netG
end

function defineG(input_nc, output_nc, ngf)
    local netG = nil
    if     opt.which_model_netG == "encoder_decoder" then netG = defineG_encoder_decoder(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet" then netG = defineG_unet(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet_swap" then netG = defineG_unet_swap(input_nc, output_nc, ngf, 1)
    elseif opt.which_model_netG == "unet_128" then netG = defineG_unet_128(input_nc, output_nc, ngf)
    else error("unsupported netG model")
    end
   
    netG:apply(weights_init)

    return netG
end

function defineD(input_nc, output_nc, ndf)
    local netD = nil
    if opt.condition_GAN==1 then
        input_nc_tmp = input_nc
    else
        input_nc_tmp = 0 -- only penalizes structure in output channels 
    end
    
    if     opt.which_model_netD == "basic" then netD = defineD_basic(input_nc_tmp, output_nc, ndf)
    elseif opt.which_model_netD == "n_layers" then netD = defineD_n_layers(input_nc_tmp, output_nc, ndf, opt.n_layers_D)
    else error("unsupported netD model")
    end
    
    netD:apply(weights_init)
    
    return netD
end

local mask_global = torch.ByteTensor(opt.batchSize, opt.fineSize, opt.fineSize)

-- load pretrained model
print('Loading pretrained model...')
local netG_pre = util.load(opt.pretrained, opt)


-- init artificial swapX to avoid error
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



local mask_global2_fake = create_gMask(pattern, mask_global, MAX_SIZE, opt):squeeze()
mask_global2_fake = mask_global2_fake:squeeze()

local kbar_fake = torch.CudaTensor(1,4096,64,64)
local k_ind_fake = torch.CudaTensor(4096)

-- local mask_global2_fake, kbar_fake, k_ind_fake = nil, nil, nil


swap2 = nn.innerSwap_stage2('innerSwap_64', kbar_fake ,k_ind_fake,mask_global2_fake, 2, 3, opt.threshold, 6, opt.type)


-- load saved models and finetune
if opt.continue_train == 1 then
   print('loading previously trained netG...')
   netG = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), opt)
   print('loading previously trained netD...')
   netD = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), opt)
else
  print('define model netG...')
  netG = defineG(input_nc, output_nc, ngf)
  print('define model netD...')
  netD = defineD(input_nc, output_nc, ndf)
end

print('netD...')
print(netD)


local criterion = nn.BCECriterion()
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
----------------------------------------------------------------------------
local real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
local real_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_B1 = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local real_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local fake_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local errD, errG, errL1 = 0, 0, 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------

if opt.gpu > 0 then
   print('transferring to gpu...')
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   real_A = real_A:cuda();
   real_B = real_B:cuda(); fake_B = fake_B:cuda();
   real_AB = real_AB:cuda(); fake_AB = fake_AB:cuda();
   if opt.cudnn==1 and opt.continue_train == 0 then
      netG = util.cudnn(netG); netD = util.cudnn(netD);
      netG_pre = util.cudnn(netG_pre)
   end
   netD:cuda(); netG:cuda(); criterion:cuda(); criterionAE:cuda();
   print('done')
else
    print('running model on CPU')
end
print('netG_pre****************')
printNet(netG_pre)
print('netG****************')
printNet(netG)
--******************* New add *******************
-- re-init netG with enc-dec
netG = layinit.encDec_init_Unet(netG_pre, netG, false)  -- false means don't copy the latter part


local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()



if opt.display then disp = require 'display' end



-- size should be num+2, as it contains '.' and '..'
local feature_all =  paths.dir(paths.concat(opt.DATA_ROOT, opt.phase))
table.remove(feature_all,1)
table.remove(feature_all,1)

-- features_all_rv is a table with keys the names of images and value 0/1 
-- separately indicating whether it has been extracted before.
local feature_all_rv = {}
local cnt_all = 0
for _, v in pairs(feature_all) do
    cnt_all = cnt_all+1
    feature_all_rv[v] = 0
end


data_path = {}
function createRealFake()

    -- load real
    data_tm:reset(); data_tm:resume()
    local real_data
    real_data, data_path = data:getBatch()

    local file_name = util.basename_batch(data_path)
    local str_len = string.len(file_name[1])
    local real_name = file_name[1]:sub(1,str_len-4)
    
    -- It is ugly.. later maybe it can be done in ugly_load
    while feature_all_rv[real_name] == 1 do
        print(string.format('Reload images: %s', real_name))
        real_data, data_path = data:getBatch()
        file_name = util.basename_batch(data_path)
        str_len = string.len(file_name[1])
        real_name = file_name[1]:sub(1,str_len-4)
    end    

    feature_all_rv[real_name] = 1  --assign it



    real_data = real_data:cuda()  -- This should be omitted.
    data_tm:stop()

	local temp_mask = torch.ByteTensor(mask_global:size())
	temp_mask:copy(real_data[{ {}, 7, {}, {} }]) 
	mask_global = 1 - temp_mask/255

-- mask_global must be byteTensor    
    real_A:copy(real_data[{ {}, idx_A, {}, {} }])
    real_A[{{},{1},{},{}}][mask_global] = 2*117.0/255.0 - 1.0
    real_A[{{},{2},{},{}}][mask_global] = 2*104.0/255.0 - 1.0
    real_A[{{},{3},{},{}}][mask_global] = 2*123.0/255.0 - 1.0

    real_B:copy(real_data[{ {}, idx_B, {}, {} }])

    mask_global = mask_global:squeeze()

    -- real_A is input(we use GAN preliminary filled images), and real_B is groundTruth
    if opt.condition_GAN==1 then
        real_AB = torch.cat(real_A,real_B,2)
    else
        real_AB = real_B -- unconditional GAN, only penalizes structure in B
    end

    -- get the latent of pretrained model and then swap.
    netG_pre:forward(real_A)
    local style_latent2 = nil
    if torch.type(netG_pre) == 'nn.gModule' then
        for indexNode, node in pairs(netG_pre.forwardnodes) do
            if indexNode == 41 then
                if node.data.module then
                    assert(torch.type(node.data.module) == 'cudnn.ReLU' or torch.type(node.data.module) == 'nn.ReLU',
                        string.format('Expecting get ReLU, yet got %s',torch.type(node.data.module)))
                    style_latent2 = node.data.module.output:clone()
                end          
            end
        end
    elseif torch.type(netG_pre) == 'nn.Sequential' then
        error('haven\'t completed yet!')
    else
        error('bad model')
    end

    local mask = cal_feat_mask(mask_global, 2, opt.threshold)

    local conv_enc, _, _, _, flag = NonparametricPatchAutoencoderFactory.buildAutoencoder(
                                                                    style_latent2:squeeze(), mask, 1, 1, 1, false, false, true)
    local maxcoor = nn.MaxCoord():cuda()
    conv_enc:cuda()
    local tmp1 = conv_enc:forward(style_latent2) 
    local kbar, k_ind = maxcoor:forward(tmp1)  
    k_ind = k_ind:cuda()

    -- kbar is 4 dimensions
    -- calulate the real kbar 
    local real_npatches = kbar:size(2) + torch.sum(flag)
    local kbar_c = kbar:size(2)
    local kbar_h = kbar:size(3)
    local kbar_w = kbar:size(4)
    kbar = torch.cat(kbar:zero(), torch.zeros(1, real_npatches - kbar_c, kbar_h, kbar_w):typeAs(kbar), 2)


    for i = 1, kbar_h do
        for j = 1, kbar_w do
            local indx = (i-1)*kbar_w + j
            local non_r_ch = k_ind[indx]
            -- get the corrected channel, set 0 to 1
            local correct_ch = non_r_ch + torch.sum(flag[{{1, indx}}])
            kbar[{{},{correct_ch},{i},{j}}] = 1   
            k_ind[indx] = correct_ch  
        end
    end
    swap2 = nn.innerSwap_stage2('innerSwap_64', kbar, k_ind, mask_global, 2, 3, opt.threshold, 6, opt.type)

    netG:replace(function(module)
        if  module.cls == 'innerSwap_64' then
            return swap2:cuda()
        else
            return module
        end
    end)
-- The swap layer index is right! Have checked!


    fake_B = netG:forward(real_A)


  
    if opt.condition_GAN==1 then
        fake_AB = torch.cat(real_A,fake_B,2)
    else
        fake_AB = fake_B -- unconditional GAN, only penalizes structure in B
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
    
    -- Fake
    -- train netD with (fake_AB, fake_label)
    local output = netD:forward(fake_AB)
    label:fill(fake_label)
    local errD_fake = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(fake_AB, df_do)
    
    errD = (errD_real + errD_fake)/2
    
    return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersG:zero()
    
    -- GAN loss
    local df_dg = torch.zeros(fake_B:size())
    if opt.gpu>0 then 
    	df_dg = df_dg:cuda();
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
    else
        errG = 0
    end
    
    -- unary loss
    local df_do_AE = torch.zeros(fake_B:size())
    if opt.gpu>0 then 
    	df_do_AE = df_do_AE:cuda();
    end
    if opt.use_L1==1 then
       errL1 = criterionAE:forward(fake_B, real_B)
       df_do_AE = criterionAE:backward(fake_B, real_B)
    else
        errL1 = 0
    end

    netG:backward(real_A, df_dg + df_do_AE:mul(opt.lambda))
    -- check_gradient(netG, 41, real_A, 1e-5)
    -- check_gradient(netG, 46, real_A, 1e-5)

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
    if not util.containsValue({"errG", "errD", "errL1"}, v) then 
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
        if opt.use_GAN==1 then optim.adam(fDx, parametersD, optimStateD) end


        -- (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        optim.adam(fGx, parametersG, optimStateG)

        -- need improve, it is ugly
        local all_feat_num = paths.dir(paths.concat(opt.features_dir,'layer64'))
        -- becuse '.' and '..' are included if using paths.dir
        if #all_feat_num >= data:size()+2 then
            print('All features has been extracted')
            print(string.format('The total num of feature maps are %d, please check it!',#all_feat_num - 2))
            os.exit()
        end

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
                        if image_out==nil then image_out = torch.cat(util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()),3)
                        else image_out = torch.cat(image_out, torch.cat(util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()),3), 2) end
                    end
                end
            end
            image.save(paths.concat(opt.checkpoints_dir,  opt.name , counter .. '_train_res.png'), image_out)
            
            opt.serial_batches=serial_batches
        end
        
        -- logging and display plot
        if counter % opt.print_freq == 0 then
            local loss = {errG=errG and errG or -1, errD=errD and errD or -1, errL1=errL1 and errL1 or -1}
            local curItInBatch = ((i-1) / opt.batchSize)
            local totalItInBatch = math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize)
            print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                    .. '  Err_G: %.4f  Err_D: %.4f  ErrL1: %.4f'):format(
                     epoch, curItInBatch, totalItInBatch,
                     tm:time().real / opt.batchSize, data_tm:time().real / opt.batchSize,
                     errG, errD, errL1))
           
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
          end
      end
    end
    
    
    parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
    parametersG, gradParametersG = nil, nil
    
    if epoch % opt.save_epoch_freq == 0 then
        torch.save(paths.concat(opt.checkpoints_dir, opt.name,  epoch .. '_net_G.t7'), netG:clearState())
        torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_D.t7'), netD:clearState())
    end
    
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
    parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
    parametersG, gradParametersG = netG:getParameters()
end
