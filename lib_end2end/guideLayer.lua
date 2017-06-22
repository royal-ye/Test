

local GuideLayer, parent = torch.class('nn.GuideLayer', 'nn.Module')

function GuideLayer:__init(swap_sz, stride, mask_thred)
    parent.__init(self)
    self.swap_sz = swap_sz
    self.stride = stride
    self.mask_thred = mask_thred
    
end

-- input should be 4 dimensions.
-- the first part is 1*C*H*W --first is the latent , the latter part is 1*1*H*W--latent is mask
function GuideLayer:updateOutput(input)
    assert(input:nDimension() == 4)  
    self.c_real = input:size(2)
    self.c = self.c_real - 1
    local data = input:narrow(2, 1, self.c):clone():squeeze()  
    local mask = input:narrow(2, self.c_real, 1):clone():squeeze() 

    local conv_enc, _, _, _, flag = NonparametricPatchAutoencoderFactory.buildAutoencoder(
        data:squeeze(), mask:squeeze(), self.swap_sz, self.stride, self.mask_thred, false, false, true)  --构造非mask区域的conv_enc
    local maxcoor = nn.MaxCoord():cuda()
    conv_enc:cuda()
    local tmp1 = conv_enc:forward(data) 
    local kbar, k_ind = maxcoor:forward(tmp1)  
    k_ind = k_ind:cuda()
    kbar = kbar:cuda()
    -- calulate the real kbar 
    local real_npatches = kbar:size(1) + torch.sum(flag)
    local kbar_c = kbar:size(1)
    local kbar_h = kbar:size(2)
    local kbar_w = kbar:size(3)
    kbar = torch.cat(kbar, torch.zeros(real_npatches - kbar_c, kbar_h, kbar_w):typeAs(kbar), 1)

    for i = 1, kbar_h do
        for j = 1, kbar_w do
            local indx = (i-1)*kbar_w + j
            -- get ori non-correct channel, set 1 to 0
            local non_r_ch = k_ind[indx]
            kbar[{{non_r_ch},{i},{j}}] = 0
            -- get the corrected channel, set 0 to 1
            local correct_ch = non_r_ch + torch.sum(flag[{{1, indx}}])
            kbar[{{correct_ch},{i},{j}}] = 1   
            k_ind[indx] = correct_ch  
        end
    end 
    -- resize k_ind:1*(H*W) to 1*1*H*W
    k_ind = k_ind:view(1,1,kbar_h, kbar_w) 
    kbar = kbar:view(1,(#kbar)[1], (#kbar)[2], (#kbar)[3])
    mask = mask:view(1,1,(#mask)[1],(#mask)[2])

    self.output = torch.cat({mask, k_ind, kbar}, 2)  
    return self.output
end


-- It can be set to anything, it doesn't matter.
function GuideLayer:updateGradInput(input, gradOutput)
    self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
    self.gradInput = nn.utils.recursiveFill(self.gradInput, 0)
    return self.gradInput

end


function GuideLayer:clearState()
    self.output = self.output.new()
    self.gradInput = self.gradInput.new()

end

