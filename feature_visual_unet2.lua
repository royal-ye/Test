-- usage example: DATA_ROOT=/path/to/data/ which_direction=BtoA name=expt1 th train.lua 
--
-- code derived from https://github.com/soumith/dcgan.torch
--

require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util/util.lua')
--require 'lib-new/MaxCoord'
require 'image'
require 'models'
require 'math'
require 'lib/NonparametricPatchAutoencoderFactory'
require 'lib/MaxCoord'
require 'lib/InstanceNormalization'
require 'lib/innerSwap'
require 'lib/myUtils'

opt = {
   name = 'feature_visual',              -- name of the experiment, should generally be passed on the command line
   --DATA_ROOT = './datasets/fix-3-mask-s-label',         -- path to images (should have subfolders 'train', 'val', etc)
   DATA_ROOT = './datasets/test-feature-visual',         -- path to images (should have subfolders 'train', 'val', etc)
   batchSize = 1,          -- # images in batch
   loadSize = 256,         -- scale images to this size
   fineSize = 256,         --  then crop to this size
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   input_nc = 6,           -- #  of input image channels
   input_nc_net = 3,           -- #  of input image channels
   output_nc = 3,          -- #  of output image channels
   niter = 10000, --200        -- #  of iter at starting learning rate
   lr = 0.15,            -- initial learning rate for adam
   beta1 = 0.2,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   flip = 0,               -- if flip the images for data argumentation
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   display_plot = 'errD',    -- which loss values to plot over time. Accepted values include a comma seperated list of: errL1, errG, and errD
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   
   which_direction = 'AtoB',    -- AtoB or BtoA
   phase = 'train',             -- train, val, test, etc
   preprocess = 'regular',      -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
   nThreads = 10,                -- # threads for loading data
   save_epoch_freq = 10,  --20       -- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
   save_latest_freq = 5000, --5000    -- save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
   print_freq = 100, --20            -- print the debug information every print_freq iterations
   display_freq = 5, --100         -- display the current results every display_freq iterations
   save_display_freq = 100,--1000    -- save the current display of results every save_display_freq_iterations
   continue_train=0,            -- if continue training, load the latest model: 1: true, 0: false
   serial_batches = 0,          -- if 1, takes images in order to make batches, otherwise takes them randomly
   serial_batch_iter = 1,       -- iter into serial image list
   checkpoints_dir = './checkpoints_visual', -- models are saved here
   cudnn = 1,                         -- set to 0 to not use cudnn
   condition_GAN = 0,                 -- set to 0 to use unconditional discriminator
   use_GAN = 1,                       -- set to 0 to turn off GAN term
   use_L1 = 1,                        -- set to 0 to turn off L1 term
   which_model_netD = 'basic', -- selects model to use for netD
   which_model_netG = 'unet',  -- selects model to use for netG
   n_layers_D = 0,             -- only used if which_model_netD=='n_layers'
   lambda = 1,               -- weight on L1 term in objective
   SwapKernelSize = 3,       -- kernel size of swap
   swap_layer = '20000000',           -- swap the layer 1,2,3,4,5,6,7,8
   AreaThreshold = 0.5,               --Theshold for computing feature mask
   swap_stride = 1,                   -- swap stride
   end_layer = 4,
   content_weight = 13,

}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)


------------------------------------init param-----------------------------------------
local input_nc = opt.input_nc_net
local output_nc = opt.output_nc
local ngf = opt.ngf

if opt.display then disp = require 'display' end
-- translation direction

local idx_A = nil
local idx_B = nil
idx_A = {1, input_nc} -- 123通道为input图 content图
idx_B = {input_nc+1, input_nc*2} -- 456通道为style图
idx_C= {input_nc+output_nc+1,input_nc*2+output_nc}  -- 789通道为groundtruth

if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

local errD = 0;

-------------------------------------- create data loader---------------------------------
local data_loader = paths.dofile('data/dataC.lua')
print('#threads...' .. opt.nThreads)

local data = data_loader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())

tmp_d, tmp_paths = data:getBatch()



---------------------------------------define network--------------------------------------------------

local netG = nil
netG = util.load('./checkpoints/tec_swap64_6-16/latest_net_G.t7', opt)

local function weights_init(netD,end_layer)
    local j=1
    for  i = 1, end_layer do
        local layer = netG:get(i)
        local tn = torch.typename(layer)
        if tn == 'cudnn.SpatialConvolution' or tn=='cudnn.SpatialFullConvolution' or tn == 'nn.InstanceNormalization' then
			if torch.typename(netD.modules[j]) == 'nn.LeakyReLU' then
			j=j+1
			end
			print(netD.modules[j])
			print(netG.modules[i])
			print(i,j)
            netD.modules[j].weight = netG.modules[i].weight:clone()
            netD.modules[j].bias = netG.modules[i].bias:clone()
			j=j+1
			
        end
    end
--assert(1==2)
    return netD
end

function defineFV(input_nc, output_nc, ngf,end_layer)

    -- input is (nc) x 256 x 256
    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1) -- 1

    if end_layer == 1 then
        local e=e1 - nn.LeakyReLU(0.2, true)
        return nn.gModule({e1},{e})
    end

    -- input is (ngf) x 128 x 128
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2) - nn.LeakyReLU(0.2, true)--2 3 4

    if end_layer == 4 then
        return nn.gModule({e1},{e2})
    end

    -- input is (ngf * 2) x 64 x 64
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4) -- nn.LeakyReLU(0.2, true)--5 6 7

    if end_layer == 7 then
        return nn.gModule({e1},{e3})
    end

    -- input is (ngf * 4) x 32 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) -- 8 9 10

    if end_layer == 10 then
        return nn.gModule({e1},{e4})-- nn.LeakyReLU(0.2, true)
    end

    -- input is (ngf * 8) x 16 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) -- 11 12 13

    if end_layer == 13 then
        return nn.gModule({e1},{e5 - nn.LeakyReLU(0.2, true)})
    end

    -- input is (ngf * 8) x 8 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) --14 15 16

    if end_layer == 16 then
        return nn.gModule({e1},{e6- nn.LeakyReLU(0.2, true)})
    end

    -- input is (ngf * 8) x 4 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) --17 18 19

    if end_layer == 19 then
        return nn.gModule({e1},{e7})
    end

    -- input is (ngf * 8) x 2 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- nn.SpatialBatchNormalization(ngf * 8) -- 20 21 
    -- input is (ngf * 8) x 1 x 1
    if end_layer == 21 then
        return nn.gModule({e1},{e8})
    end
    local d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5) -- 22 23 24 25

    if end_layer == 25 then
        return nn.gModule({e1},{d1_})
    end

    -- input is (ngf * 8) x 2 x 2
    local d1 = {d1_,e7} - nn.JoinTable(2) --26
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5) -- 27 28 29 30

    if end_layer == 30 then
        return nn.gModule({e1},{d2_})
    end

    -- input is (ngf * 8) x 4 x 4
    local d2 = {d2_,e6} - nn.JoinTable(2) --31
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5) --  32 33 34 35

    if end_layer == 35 then
        return nn.gModule({e1},{d3_})
    end

    -- input is (ngf * 8) x 8 x 8
    local d3 = {d3_,e5} - nn.JoinTable(2) --36
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) -- 37 38 39

    if end_layer == 39 then
        return nn.gModule({e1},{d4_- nn.LeakyReLU(0.2, true)})
    end

    -- input is (ngf * 8) x 16 x 16
    local d4 = {d4_,e4} - nn.JoinTable(2) --40
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4) -- 41 42 43

    if end_layer == 43 then
        return nn.gModule({e1},{d5_})
    end

    -- input is (ngf * 4) x 32 x 32
    local d5 = {d5_,e3} - nn.JoinTable(2) -- 44
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2) -- 45 46 47

    if end_layer == 47 then
        return nn.gModule({e1},{d6_})
    end

    -- input is (ngf * 2) x 64 x 64
    local d6 = {d6_,e2} - nn.JoinTable(2) -- 48
    local d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)  -- 49 50 51
    
    if end_layer == 51 then
        return nn.gModule({e1},{d7_})
    end

    error('not valid end layer')
    

end

local netFV = nil

netFV = defineFV(opt.input_nc_net,opt.output_nc,opt.ngf,opt.end_layer)

netFV = util.cudnn(netFV); 

netFV = weights_init(netFV,opt.end_layer)


for i=1,#netG.modules do
    local layer = netG:get(i)
    local tn = torch.typename(layer)
    print(i,tn)
end
for i=1,#netFV.modules do
    local layer = netFV:get(i)
    local tn = torch.typename(layer)
    print(i,tn)
end
----------------------------------------------------------------------------
local real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
local real_B = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
local real_C = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
local FeatureGroundtruth = nil;


local errD= 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------------------
-- local criterion = nn.AbsCriterion();   --abs 损失函数
local criterion = nn.MSECriterion()      --mse 损失函数

optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
-- ----------------------------------------------------------------------------

if opt.gpu > 0 then
   print('transferring to gpu...')
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   real_A = real_A:cuda();
   real_B = real_B:cuda(); 
   real_C = real_C:cuda();
   
  
   if opt.cudnn==1 then
      netG = util.cudnn(netG);
      netFV = util.cudnn(netFV);
   end
   netG:cuda();criterion:cuda(); netFV:cuda(); 
   print('done')
else
	print('running model on CPU')
end

-- local parametersFV, gradParametersFV = netFV:getParameters()


local imageB_ = image.load('./datasets/test-feature-visual/train/3.png')
local imageB = imageB_[{{},{},{1,256}}]
-- -------------------------------generate data image-------------------------------------------------
function createRealFake()
    data_tm:reset(); data_tm:resume()
    local real_data, data_path = data:getBatch()
    data_tm:stop()
    real_A:copy(real_data[{ {}, idx_A, {}, {} }]) --1*3*256*256 -1 到 1 之间
    real_B:copy(real_data[{ {}, idx_B, {}, {} }]) -- 为mask 范围为 -1 和 1 4D
    real_C:copy(real_data[{ {}, idx_C, {}, {} }])
    --image_B = real_A:clone()
    netG:forward(real_A)
    FeatureGroundtruth_tmp=netG.modules[48].output:clone()--
	print(FeatureGroundtruth_tmp:size())
	print(netG.modules[48])--this is swaped feature
	--assert(1==2)
	--print()
	FeatureGroundtruth = FeatureGroundtruth_tmp[{{},{1,128},{},{}}]
	print(FeatureGroundtruth:size())
end

createRealFake()
local img = nil;
img = torch.randn(real_A:size()):float():mul(0.001):cuda()

--  img=real_B:clone();

-- -- create closure to evaluate f(X) and df/dX of discriminator

local function feval(x)
    --netD_8:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    -- gradParametersD8:zero();
    ----------------------------start foward 1-7 ----------------------------------------------------
    local Output = netFV:forward(x)
    errD = criterion:forward(Output, FeatureGroundtruth)
    local df = criterion:backward(Output, FeatureGroundtruth)

    local grad = netFV:updateGradInput(x,df)
    
    --print(netD_8:updateGradInput(i8,df_d8):size())
    --image.save('test.png',netD_8:forward(i8)[1])
    --image.save('testG.png',G_output[1])
   
    return errD, grad -- 1*3*256*256
end

local function test(x)
    --netD_8:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    -- gradParametersD8:zero();
    ----------------------------start foward 1-7 ----------------------------------------------------
    
    local Output = netFV:forward(x)
    
    errD = criterion:forward(Output, FeatureGroundtruth)
    local df = criterion:backward(Output, FeatureGroundtruth)

    local grad = netFV:updateGradInput(x,df)
    
    --print(netD_8:updateGradInput(i8,df_d8):size())
    --image.save('test.png',netD_8:forward(i8)[1])
    --image.save('testG.png',G_output[1])
   
    return errD, grad:mul(opt.content_weight)
end
-- ----------------------------------------------- train-------------------------------------------
local best_err = nil
paths.mkdir(opt.checkpoints_dir)
paths.mkdir(opt.checkpoints_dir .. '/' .. opt.name ..'_'..opt.end_layer)

image.save(paths.concat(opt.checkpoints_dir,  opt.name..'_'..opt.end_layer , '0_train_res.png'), util.deprocess(img[1]:float()))

-- save opt
file = torch.DiskFile(paths.concat(opt.checkpoints_dir, opt.name..'_'..opt.end_layer, 'opt.txt'), 'w')
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

local destination_position
local psnr
local psnr_max=0
local crit = nn.MSECriterion()
for epoch = 1, opt.niter do
    epoch_tm:reset()
    for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do 
        tm:reset()
        
        optim.adam(feval, img, optimStateD)
        
        --test(img)
        -- display
        --print((a:float()-b:float()):sum()..'**************************************')
        counter = counter + 1
        if counter % opt.display_freq == 0 and opt.display then
            disp.image(util.deprocess_batch(util.scaleBatch(img:float(),256,256)), {win=opt.display_id, title=opt.name .. ' input'})
            
        end
      
        -- write display visualization to disk
        --  runs on the first batchSize images in the opt.phase set
        if counter % opt.save_display_freq == 0 and opt.display then
            local image_out = util.deprocess(img[1]:float())
               
            --print(psnr)
            image.save(paths.concat(opt.checkpoints_dir,  opt.name..'_'..opt.end_layer , counter .. '_train_res.png'), image_out)
            imageA = image.load(paths.concat(opt.checkpoints_dir,  opt.name..'_'..opt.end_layer , counter .. '_train_res.png'))
             
             mse = crit:forward(imageA:float(), imageB:float())--计算PSNR指标用
                psnr = 10*math.log10(1/mse)
                if psnr > psnr_max and counter > 500 then
                psnr_max = psnr
                destination_position = counter
                end
        end
        -- logging and display plot
        if counter % opt.print_freq == 0 then
            local loss = {errD=errD and errD or -1}
            local curItInBatch = ((i-1) / opt.batchSize)
            local totalItInBatch = math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize)
            print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                    .. ' Err_D: %.4f '):format(
                     epoch, curItInBatch, totalItInBatch,
                     tm:time().real / opt.batchSize, data_tm:time().real / opt.batchSize,
                     errD))
            
            print(psnr)
           
            local plot_vals = { epoch + curItInBatch / totalItInBatch }
            for k, v in ipairs(opt.display_plot) do
              if loss[v] ~= nil then
               plot_vals[#plot_vals + 1] = loss[v] 
             end
            end

            -- update display plot
            if opt.display then
              table.insert(plot_data, plot_vals)
              plot_config.win = plot_win
              plot_win = disp.plot(plot_data, plot_config)
            end
        end

        
    end
    
    
    -- print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
    --         epoch, opt.niter, epoch_tm:time().real))


end
print("psnr:", psnr_max, "destination_position:",destination_position)
