require 'lib_end2end/NonparametricPatchAutoencoderFactory'

-- It is use in a test experiment, mainly check whether 3*3 swap size are
-- better than 1*1  for 64*64 layer


-- It is especially for stage2
-- Here just put self.kbar and self.ind to conv_dec(constructed by latterAll). 

local innerSwap_stage2, parent = torch.class('nn.innerSwap_stage2', 'nn.Module')

function innerSwap_stage2:__init(cls, kbar, k_ind, mask, conv_layers, swap_sz, threshold, mask_thred, purpose)
    parent.__init(self)

    self.cls = cls
    self.mask = mask:byte()
    self.kbar = kbar
    self.ind = k_ind
    self.conv_layers = conv_layers
    assert(mask:nDimension() == 2)
    self.swap_sz = swap_sz   -- for layer near fc, it should be 1 and layer far from fc, it may can be larger like 3
    self.threshold = threshold  -- It is for defining a clear mask in certain feature mask.
    self.mask_thred = mask_thred -- It is for defining whether patch is mask or not.
    self.purpose = purpose  -- 'data_extraction' or 'training'

end


-- Use latter non-mask region as style to swap former masked region.

function innerSwap_stage2:updateOutput(input)
    assert(input:nDimension() == 4)
    self.bz = input:size(1)
    self.c = input:size(2)
    self.h = input:size(3)
    self.w = input:size(4)
    -- 4 dimensions
    local former = input:narrow(2, 1, self.c/2)
    local latter = nil
    local new_latter = nil

    local file_name = util.basename_batch(data_path)
    local str_len = string.len(file_name[1])
    
    local real_name = file_name[1]:sub(1,str_len-4)
    local iL = string.sub(string.match(self.cls, '_%d+'),2)
    local data_name = real_name..'_'..iL..'.t7'

    if self.purpose == 'data_extraction' then
        latter = input:narrow(2, self.c/2 + 1, self.c/2)
        local mask = cal_feat_mask(self.mask, self.conv_layers, self.threshold) 

        local ex_mask = mask:byte():repeatTensor(self.bz, self.c/2 ,1,1)
        local inv_ex_mask = ex_mask:clone()

        local tmp_mask = torch.ByteTensor(inv_ex_mask:size()):fill(1)

        inv_ex_mask[torch.lt(inv_ex_mask, tmp_mask)] = 2
        inv_ex_mask[torch.eq(inv_ex_mask, tmp_mask)] = 0
        inv_ex_mask[torch.gt(inv_ex_mask, tmp_mask)] = 1


        local latter_non_mask = latter:clone()
        latter_non_mask[ex_mask:byte()] = 0  --only save non_mask region
        
        -- only accept 3 dimension, so squeeze
        _, conv_new_dec,self.centers_lt, self.centers_rb, _ = NonparametricPatchAutoencoderFactory.buildAutoencoder(
                        latter:squeeze(), mask:zero(), self.swap_sz, 1, self.mask_thred, false, false, true)
        conv_new_dec:cuda()


    -- result_tmp should be an image that masked region swapped(with latter non-mask) and non-mask(garbage)
        local result_tmp = conv_new_dec:forward(self.kbar)  
        local result_tmp_mask = result_tmp:clone()
        result_tmp_mask[inv_ex_mask] = 0
        new_latter = torch.add(result_tmp_mask, latter_non_mask)

    
        self.ex_mask = ex_mask:clone()
        self.inv_ex_mask = inv_ex_mask:clone()

        torch.save(paths.concat(opt.features_dir, 'layer'..iL, data_name),new_latter)

        torch.save(paths.concat(opt.features_dir, 'layer_ori'..iL, 'ori_'..data_name),latter)
    elseif self.purpose == 'training' then
        new_latter = torch.load(paths.concat(opt.features_dir, 'layer'..iL, data_name))
    else
        error('ambiguous purpose！')
    end
    
-- construct final self.output
    self.output = torch.cat(former, new_latter, 1):view(1, self.c, self.h, self.w)
    return self.output
end


-- LM: latter mask : mask blanket
-- LN: latter non-mask : clear non-mask image
-- PM: former mask : blur mask region
-- PN: former non-mask : non-mask blanket
-- In updateOutput, PM is swapped with LN(as style).
-- So here:

-- calculate the grad of LM using the grad of its replacing region in LN.

function innerSwap_stage2:updateGradInput(input, gradOutput)
    self.gradInput = torch.Tensor():zero():typeAs(input):resizeAs(input)
    -- The former isn't swapped, so just backprop the gradient.
    local grad_former = gradOutput[{{},{1, self.c/2},{},{}}]
    local grad_non_mask = gradOutput[{{},{self.c/2 + 1, self.c},{},{}}]:clone()
    grad_non_mask[self.ex_mask] = 0  -- save non_mask
    local grad_tmp = torch.Tensor():zero():typeAs(input):resizeAs(grad_non_mask)

    for cnt = 1, self.centers_lt:size(1) do
        local lt_C = self.centers_lt[cnt]
        local rb_C = self.centers_rb[cnt]
        
        local indS = self.ind[cnt]
        local lt_S = self.centers_lt[indS]
        local rb_S = self.centers_rb[indS]

        -- When self.swap_sz = 1, then the following equation is point-to-point gradient replacing.
        grad_tmp[{{},{},{lt_C[1],rb_C[1]},{lt_C[2],rb_C[2]}}] = 
              gradOutput[{{},{self.c/2 + 1, self.c},{lt_S[1],rb_S[1]},{lt_S[2],rb_S[2]}}]

    end
    grad_tmp[self.inv_ex_mask] = 0  --grad_tmp mask region is useful
    local grad_latter = torch.add(grad_non_mask, grad_tmp)
    self.gradInput = torch.cat(grad_former, grad_latter, 2)

    return self.gradInput

end

function innerSwap_stage2:clearState()
    self.output = self.output.new()
    self.gradInput = self.gradInput.new()

end

