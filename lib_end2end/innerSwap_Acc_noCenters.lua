require 'lib/Nonparametric_especial_noCenters'

local InnerSwap, parent = torch.class('nn.InnerSwap', 'nn.Module')

function InnerSwap:__init(cls, mask, conv_layers, latent, swap_sz, threshold, mask_thred)
    parent.__init(self)

    self.cls = cls
    self.mask = mask:byte()
    self.conv_layers = conv_layers
    self.latent = latent
    assert(self.latent:nDimension() == 4)
    self.c = latent:size(2) 
    assert(self.c % 2 == 0)
    assert(mask:nDimension() == 2)
    self.swap_sz = swap_sz   -- for layer near fc, it should be 1 and layer far from fc, it may can be larger like 3
    self.threshold = threshold  -- It is for defining a clear mask in certain feature mask.
    self.mask_thred = mask_thred -- It is for defining whether patch is mask or not.


end


-- Use latter non-mask region as style to swap former masked region.

function InnerSwap:updateOutput(input)
    assert(input:nDimension() == 4)
    self.h = input:size(3)
    self.w = input:size(4)
    local former = input:narrow(2, 1, self.c/2):clone():squeeze()   
    local latter = input:narrow(2, self.c/2 + 1, self.c/2):clone():squeeze()


    local mask = cal_feat_mask(self.mask, self.conv_layers, self.threshold)  -- Badly need update
    -- local shan_timer2 = torch.Timer()
    _, self.conv_enc, self.conv_new_dec, _ , self.flag = NonparametricPatchAutoencoderFactory.buildAutoencoder(
                                                                    latter, mask, self.swap_sz, 1, self.mask_thred, false, false, true)
    -- print('Time2: '..shan_timer2:time().real..' seconds') 
    -- shan_timer2:stop()
    self.maxcoor = nn.MaxCoord():cuda()
    self.conv_enc:cuda()  -- it is conv_enc_nonMask

    -- local shan_timer3 = torch.Timer()
    local ex_mask = mask:byte():repeatTensor(self.c/2 ,1,1)
    local inv_ex_mask = ex_mask:clone()

    local tmp_mask = torch.ByteTensor(inv_ex_mask:size()):fill(1)

    inv_ex_mask[torch.lt(inv_ex_mask, tmp_mask)] = 2
    inv_ex_mask[torch.eq(inv_ex_mask, tmp_mask)] = 0
    inv_ex_mask[torch.gt(inv_ex_mask, tmp_mask)] = 1

-- here latter mask region should be replaced with latter non-masked region. 
    local tmp1 = self.conv_enc:forward(former) 
    local latter_non_mask = latter:clone()
    latter_non_mask[ex_mask:byte()] = 0  --only save non_mask region
    self.kbar, self.ind = self.maxcoor:forward(tmp1)  


    -- -- Then using the latter as content to construct swap(only use the dec part)
    -- -- Use the all-patches, as the conv_new_dec should contain 196 patches.
    -- _, self.conv_new_dec, _, _, _, _ = NonparametricPatchAutoencoderFactory.buildAutoencoder(
    --                 latter, mask:zero(), self.swap_sz, 1, self.mask_thred, false, false, true)
    self.conv_new_dec:cuda()

-- calulate the real self.kbar and real self.ind
    local real_npatches = self.kbar:size(1) + torch.sum(self.flag)

    local kbar_c = self.kbar:size(1)
    local kbar_h = self.kbar:size(2)
    local kbar_w = self.kbar:size(3)
    self.kbar = torch.cat(self.kbar, torch.zeros(real_npatches - kbar_c, kbar_h, kbar_w):typeAs(self.kbar), 1)

    for i = 1, kbar_h do
        for j = 1, kbar_w do
            local indx = (i-1)*kbar_w + j
            -- get ori non-correct channel, set 1 to 0
            local non_r_ch = self.ind[indx]
            self.kbar[{{non_r_ch},{i},{j}}] = 0
            -- get the corrected channel, set 0 to 1
            local correct_ch = non_r_ch + torch.sum(self.flag[{{1, indx}}])

            self.kbar[{{correct_ch},{i},{j}}] = 1
            self.ind[indx] = correct_ch     
        end
    end

-- result_tmp should be an image that masked region swapped(with latter non-mask) and non-mask(garbage)
    local result_tmp = self.conv_new_dec:forward(self.kbar)  
    local result_tmp_mask = result_tmp:clone()
    result_tmp_mask[inv_ex_mask] = 0
    local new_latter = torch.add(result_tmp_mask,latter_non_mask)
 
    self.ex_mask = ex_mask:clone()
    self.inv_ex_mask = inv_ex_mask:clone()
-- construct final self.output
    self.output = torch.cat(former, new_latter, 1):view(1, self.c, self.h, self.w)
    -- print('Time3: '..shan_timer3:time().real..' seconds') 
    return self.output
end


-- LM: latter mask : mask blanket
-- LN: latter non-mask : clear non-mask image
-- PM: former mask : blur mask region
-- PN: former non-mask : non-mask blanket
-- In updateOutput, PM is swapped with LN(as style).
-- So here:

-- calculate the grad of LM using the grad of its replacing region in LN.

-- 这是标准的！！！！！！！！！！
function InnerSwap:updateGradInput(input, gradOutput)
    self.gradInput = torch.Tensor():zero():typeAs(input):resizeAs(input)
    -- The former isn't swapped, so just backprop the gradient.
    local grad_former = gradOutput[{{},{1, self.c/2},{},{}}]
    local grad_non_mask = gradOutput[{{},{self.c/2 + 1, self.c},{},{}}]:clone()
    grad_non_mask[self.ex_mask] = 0  -- save non_mask
    local grad_tmp = torch.Tensor():zero():typeAs(input):resizeAs(grad_non_mask)

    for cnt = 1, centers_lt:size(1) do
        local lt_C = centers_lt[cnt]
        local rb_C = centers_rb[cnt]
        
        local indS = self.ind[cnt]
        local lt_S = centers_lt[indS]
        local rb_S = centers_rb[indS]

        -- When self.swap_sz = 1, then the following equation is point-to-point gradient replacing.
        grad_tmp[{{},{},{lt_C[1],rb_C[1]},{lt_C[2],rb_C[2]}}] = 
              gradOutput[{{},{self.c/2 + 1, self.c},{lt_S[1],rb_S[1]},{lt_S[2],rb_S[2]}}]

    end
    grad_tmp[self.inv_ex_mask] = 0  --grad_tmp mask region is useful
    local grad_latter = torch.add(grad_non_mask, grad_tmp)
    self.gradInput = torch.cat(grad_former, grad_latter, 2)

    return self.gradInput

end


function InnerSwap:clearState()
    self.output = self.output.new()
    self.gradInput = self.gradInput.new()

end

