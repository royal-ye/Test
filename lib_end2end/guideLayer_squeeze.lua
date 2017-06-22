
local GuideLayer, parent = torch.class('nn.GuideLayer', 'nn.Module')

function GuideLayer:__init(swap_sz, stride, mask_thred)
    parent.__init(self)
    self.swap_sz = swap_sz
    self.stride = stride
    self.mask_thred = mask_thred
    
end

-- input should be 4 dimensions.
-- the first part is 1*C*H*W and the latter part is 1*1*H*W
function GuideLayer:updateOutput(input)
    assert(input:nDimension() == 4)  
    self.c_real = input:size(2)
    self.c = self.c_real - 1
    local data = input:narrow(2, 1, self.c)   -- just use the same storage of input is fine, so we don't use input anymore.
    local mask = input:narrow(2, self.c_real, 1)

    local conv_enc, _, _, _, flag = NonparametricPatchAutoencoderFactory.buildAutoencoder(
        data:squeeze(), mask:squeeze(), self.swap_sz, self.stride, self.mask_thred, false, false, true)
    local maxcoor = nn.MaxCoord():cuda()
    conv_enc:cuda()
    local tmp1 = conv_enc:forward(data) 
    local k_ind = maxcoor:forward(tmp1) 
    local k_ind_h, k_ind_w = k_ind:size(3), k_ind:size(4)
    -- print(string.format('mask area ratio %f:',torch.sum(flag)/k_ind_h/k_ind_w))

    -- calulate the real kbar 
    -- local real_npatches = kbar:size(1) + torch.sum(flag)
    local real_npatches = k_ind_h * k_ind_w -- It should excatly equals another calculation method.

    local indx
    for i = 1, k_ind_h do
        for j = 1, k_ind_w do
            indx = (i-1)*k_ind_w + j
            -- get ori non-correct channel, set 1 to 0
            local non_r_ch = k_ind[{{1},{1},{i},{j}}]

            -- get the corrected channel, set 0 to 1
            local correct_ch = non_r_ch + torch.sum(flag[{{1, indx}}])
  
            k_ind[{{1},{1},{i},{j}}] = correct_ch  
        end
    end 

    self.output = torch.cat({mask, k_ind}, 2)  
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

