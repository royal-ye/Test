
require 'lib_end2end/NonparametricPatchAutoencoderFactory'
local InnerSwap_unetPart, parent = torch.class('nn.InnerSwap_unetPart', 'nn.Module')

function InnerSwap_unetPart:__init(cls, swap_sz, threshold, mask_thred)
    parent.__init(self)

    self.cls = cls
    self.swap_sz = swap_sz   -- for layer near fc, it should be 1 and layer far from fc, it may can be larger like 3
    self.threshold = threshold  -- It is for defining a clear mask in certain feature mask.
    self.mask_thred = mask_thred -- It is for defining whether patch is mask or not.

end


-- Use latter non-mask region as style to swap former masked region.
-- when it is before the one to the last layer(after spatial size 64*64, H=W=64)
-- Input: data:  1*c*H*W  1*(2*2*64)*H*W      
--        mask:1*1*H*W       
--        k_ind: 1*1*H*W
--        kbar, 1*(H*W)*H*W  1*4096*H*W
-- when it is before the last layer(after spatial size 128*128, H=W=128)
-- Input: data:  1*c*H*W  1*(2*64)*H*W
--        mask:1*1*H*W       
--        k_ind: 1*1*H*W
--        kbar, 1*(H*W)*H*W  1*(128*128)*H*W

-- When it comes to spatial size 128, I think it will overfill.
-- Here I just use `cls` to get a quick fix. Maybe need update later.
function InnerSwap_unetPart:updateOutput(input)
    assert(input:nDimension() == 4)
    local former, latter = nil, nil
    self.c_real = input:size(2)		
    if self.cls == 'innerSwap_64' then  -- need inprove
        self.kbar_dim = 64*64
    elseif self.cls == 'innerSwap_128' then
        self.kbar_dim = 128*128
    end

    self.c = self.c_real - self.kbar_dim - 2 --why
    local former = input:narrow(2, 1, self.c/2):clone():squeeze()   
    local latter = input:narrow(2, self.c/2 + 1, self.c/2):clone():squeeze()
    self.mask = input:narrow(2, self.c+1, 1):clone():squeeze()
    self.ind = input:narrow(2, self.c+2, 1):clone():squeeze():view(-1)  --vectorized it
    self.kbar = input:narrow(2, self.c+3, self.kbar_dim):squeeze()

    self.h = input:size(3)
    self.w = input:size(4)


    local ex_mask = self.mask:byte():repeatTensor(self.c/2 ,1,1)--不知道啥意思
    local inv_ex_mask = ex_mask:clone()

    local tmp_mask = torch.ByteTensor(inv_ex_mask:size()):fill(1)

    inv_ex_mask[torch.lt(inv_ex_mask, tmp_mask)] = 2
    inv_ex_mask[torch.eq(inv_ex_mask, tmp_mask)] = 0
    inv_ex_mask[torch.gt(inv_ex_mask, tmp_mask)] = 1


    local latter_non_mask = latter:clone()
    latter_non_mask[ex_mask:byte()] = 0  --only save non_mask region
    
    -- mask is useless, we can just use a zeros to replace.
    -- So latter we can kick out the mask input.
    _, self.conv_new_dec,self.centers_lt, self.centers_rb, _ = NonparametricPatchAutoencoderFactory.buildAutoencoder(
                    latter, self.mask:byte():zero(), self.swap_sz, 1, self.mask_thred, false, false, true)
    self.conv_new_dec:cuda()

-- result_tmp should be an image that masked region swapped(with latter non-mask) and non-mask(garbage)
    local result_tmp = self.conv_new_dec:forward(self.kbar)  
    local result_tmp_mask = result_tmp:clone()
    result_tmp_mask[inv_ex_mask] = 0
    local new_latter = torch.add(result_tmp_mask, latter_non_mask)
 
    self.ex_mask = ex_mask:clone()
    self.inv_ex_mask = inv_ex_mask:clone()
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

function InnerSwap_unetPart:updateGradInput(input, gradOutput)
    self.gradInput = torch.Tensor():zero():typeAs(input):resizeAs(input)
    -- The former isn't swapped, so just backprop the gradient.
    local grad_former = gradOutput[{{},{1, self.c/2},{},{}}]:clone()
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

    -- zeroing extra dimension
    local extra_dim = self.kbar_dim + 2
    local grad_extra = torch.zeros(1, extra_dim, self.h, self.w):typeAs(input)

    self.gradInput = torch.cat({grad_former, grad_latter, grad_extra}, 2)

    return self.gradInput

end

function InnerSwap_unetPart:clearState()
    self.output = self.output.new()
    self.gradInput = self.gradInput.new()

end

