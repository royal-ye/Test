-- require 'cunn'
-- require 'cutorch'
-- require 'torch'
-- require 'lib/myUtils'
require 'lib/NonparametricPatchAutoencoderFactory'


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

    -- self.centers are totally logged, so they can be used in backward to find out the
    -- exact position.
    local mask = cal_feat_mask(self.mask, self.conv_layers, self.threshold)  -- Badly need update
    self.conv_enc, self.conv_dec, self.centers_lt, self.centers_rb, self.flag = NonparametricPatchAutoencoderFactory.buildAutoencoder(
                                                                    latter, mask, self.swap_sz, 1, self.mask_thred, false, false, true)
    self.maxcoor = nn.MaxCoord():cuda()
    self.conv_enc:cuda()

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


    -- Then using the latter as content to construct swap(only use the dec part)
    -- Use the all-patches, as the conv_new_dec should contain 196 patches.
    _, self.conv_new_dec, _, _, _, _ = NonparametricPatchAutoencoderFactory.buildAutoencoder(
                    latter, mask:zero(), self.swap_sz, 1, self.mask_thred, false, false, true)
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
    local new_latter = result_tmp_mask + latter_non_mask
 
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

function InnerSwap:updateGradInput(input, gradOutput)
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





-- function InnerSwap:updateOutput(input)
--     print('updating')
--     assert(input:nDimension() == 4)
--     local former = input:narrow(2, self.c/2 + 1, self.c/2):squeeze()
--     local latter = input:narrow(2, 1, self.c/2):squeeze()

--     local mask = cal_feat_mask(self.mask, self.conv_layers)
--     local ex_mask = mask:byte():repeatTensor(self.c/2 ,1,1)

--     -- need to optimize.. 
--     local inv_ex_mask = ex_mask:clone()

--     local tmp_mask = torch.ByteTensor(inv_ex_mask:size()):fill(1)

--     inv_ex_mask[torch.lt(inv_ex_mask, tmp_mask)] = 2
--     inv_ex_mask[torch.eq(inv_ex_mask, tmp_mask)] = 0
--     inv_ex_mask[torch.gt(inv_ex_mask, tmp_mask)] = 1
--     inv_ex_mask = inv_ex_mask:typeAs(input)
--     ex_mask = ex_mask:typeAs(input)
--     -- construct masked style and masked content
--     local mk_style = torch.cmul(former, inv_ex_mask)
--     local mk_content = torch.cmul(former, ex_mask)

--     -- construct swap  , self.conv_dec, self.centers_lt, self.centers_rb
--     self.conv_enc, me, self.centers_lt, self.centers_rb = NonparametricPatchAutoencoderFactory.buildAutoencoder(mk_style,3, 1, false, false, true)
--     self.maxcoor = nn.MaxCoord():cuda()
--     self.conv_enc:cuda()

-- --  expand bs dimension of mk_style, mk_content, latter
--     -- mk_style = mk_style:view(1,mk_style:size(1),mk_style:size(2),mk_style:size(3))
--     -- mk_content = mk_content:view(1,mk_content:size(1),mk_content:size(2),mk_content:size(3))


--     local tmp1 = self.conv_enc:forward(mk_content)
--     self.kbar, self.ind = self.maxcoor:forward(tmp1)
--     -- self.useless_out = self.conv_dec:forward(self.kbar)

--     -- local tm2 = torch.Timer()
--     -- _,self.conv_dec,_, _ = NonparametricPatchAutoencoderFactory.buildAutoencoder(latter, 3, 1, false, true)
--     -- self.conv_dec:cuda()


--     -- local output = self.conv_dec:forward(self.kbar)
--     -- print('tm2 time: '..tm2:time().real)

-- -- **************** Only batch_size = 1 works!************************
-- -- Are you sure to use this method ?
--     -- use swap info like self.ind and self.centers to swap the latter

-- -- Mention: since self.kbar has some problems, so figure out the mask pairs.
-- -- Non-masked region will be replace with the original latter after the global replacing.

--     local freq, fInd = torch.mode(self.ind)
--     -- torch.mode seems a bug, so use another way to calulate fInd
--     local fp = torch.eq(self.ind, freq[1])
--     fInd = fp:nonzero()

--     -- print(fInd)
--     -- Ab()

--     local output2 = torch.Tensor():typeAs(latter):resizeAs(latter):zero()

--     local tm1 = torch.Timer()
--     for cnt = 1, self.centers_lt:size(1) do
--         local lt_C = self.centers_lt[cnt]
--         local rb_C = self.centers_rb[cnt]
--         local indS = self.ind[cnt]

--         local lt_S = self.centers_lt[indS]
--         local rb_S = self.centers_rb[indS]

--         output2[{{},{(lt_C[1]+rb_C[1])/2},{(lt_C[2]+rb_C[2])/2} }]:add(latter[{{},{(lt_S[1]+rb_S[1])/2},{(lt_S[2]+rb_S[2])/2} }])
--     end
--     local divNum = getAverageTemplate(latter, 3)
-- -- replace non-swaped region with original feature.
--     for cnt = 1, fInd:size(1) do
--         local lt_C = self.centers_lt[cnt]
--         local rb_C = self.centers_rb[cnt]
--         output2[]

--     end

--     print('elapsed time: '..tm1:time().real)
--     print(#divNum)
--     output2:cdiv(divNum)
--     print('output:')
--     print(torch.sum(output))
--     print('output2:')
--     print(torch.sum(output2))
--     Ab()
-- -- construct final self.output
--     self.output = torch.cat(former, output2, 1):expandAs(input)
--     return self.output
-- end


-- It cannot be done easily as centers_lt as global variables will be wrong in netG:backward
-- I haven't figure it out.
-- -- It is hard to write the code, just a try..
-- function InnerSwap:updateGradInput2(input, gradOutput)

--     local file  = io.open('log.txt','a+')
--     file:write('swap1 updateGradInput****************..\n\n\n')
--     file:close()
--     self.gradInput = torch.Tensor():zero():typeAs(input):resizeAs(input)
--     self.centers_lt = centers_lt1_layer
--     self.centers_rb = centers_rb1_layer
--     print('swap1..gradOutput..'..torch.sum(gradOutput))
--     -- self.ind = ind1
--     print(#self.ind)
--     for cnt = 1, self.centers_lt:size(1) do
--         local left_top_C = self.centers_lt[cnt]
--         local right_bottom_C = self.centers_rb[cnt]

-- -- For inpainting, here we consider C and S are the same size, so we can consider indS as indC.
-- -- So just use centers_lt and centers_rb is fine. It saves work!
--         local indS = self.ind[cnt] -- ind for S
        
--         -- print(string.format('cnt:%d, indS:%d',cnt, indS))
--         local left_top_S = self.centers_lt[indS]
--         local right_bottom_S = self.centers_rb[indS]

--         -- dPatchC = dPatchS
--         -- self.gradInput[{ {},{}, {left_top_C[1], right_bottom_C[1]},{left_top_C[2], right_bottom_C[2]} }]:add(gradOutput[{{},{},{left_top_S[1],right_bottom_S[1]},{left_top_S[2], right_bottom_S[2]} }])

--         self.gradInput[{ {},{}, {(left_top_C[1]+right_bottom_C[1])/2},{(left_top_C[2]+right_bottom_C[2])/2} }]:add(gradOutput[{{},{},{(left_top_S[1]+right_bottom_S[1])/2},{(left_top_S[2]+right_bottom_S[2])/2} }])
        
-- --************************ Now we can just use bs = 1, for getAverageTemplate only receives input nDimension == 3
-- -- It hasn't completely finished. So just let divNum = patchSize^2.
-- --         local divNum = getAverageTemplate(dpatchC:squeeze(),0)

--     end
--     print('swap1..gradInput..'..torch.sum(self.gradInput))

--     -- self.normalize = true
--     -- if self.normalize then
--     --     self.gradInput:add(1e-7)
--     -- end
--     -- have a look
--     --assert(math.abs(torch.sum(self.gradInput))< 1e-8, 'meide!')


--     return self.gradInput

-- end


function InnerSwap:clearState()
    self.output = self.output.new()
    self.gradInput = self.gradInput.new()

end

