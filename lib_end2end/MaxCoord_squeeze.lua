--[[
Expects 3D or 4D input. Does a max over the feature channels.
--]]


-- It is quite special.
-- In this script, we don't calculate kbar, just use k_ind is enough.
-- It is because that the model is to large, so do not transfer such 
-- big data in the model. Yet, when calculate the answers we
-- have to squeeze k_ind to kbar.

-- *********** self.output equals 'self.k_ind' in the original MaxCoord!*************
-- but self.k_ind is 4 dimension. I think it is easier for later bz > 1.
local MaxCoord, parent = torch.class('nn.MaxCoord', 'nn.Module')

function MaxCoord:__init(inplace)
    parent.__init(self)
    self.inplace = inplace or false


end

function MaxCoord:updateOutput(input)
    -- if input:nDimension() == 3 then
    --     local C,H,W = input:size(1), input:size(2), input:size(3)
    --     input = input:view(1,C,H,W)
    -- end
    assert(input:nDimension()==4, 'Input must be 3D or 4D (batch).')
    local bz, c, h, w = input:size(1), input:size(2), input:size(3), input:size(4)


    local _, argmax = torch.max(input,2)
    self.output = torch.FloatTensor(bz, 1, h, w):typeAs(input):zero()

    for b=1,bz do
        for i=1,h do
            for j=1,w do
                ind = argmax[{b,1,i,j}]
                self.output[{b,1,i,j}] = ind

            end
        end
    end

    return self.output
end



-- function MaxCoord:updateOutput2(input)
--     local nInputDim = input:nDimension()
--     if input:nDimension() == 3 then
--         local C,H,W = input:size(1), input:size(2), input:size(3)
--         input = input:view(1,C,H,W)
--     end
--     assert(input:nDimension()==4, 'Input must be 3D or 4D (batch).')
--     -- print('self._type '.. self._type)
--     if self._type ~= 'torch.FloatTensor' then
--         input = input:float()
--     end
--     -- print('self._type '.. self._type)
--     -- Ab()
--     input = input:float()

--     local _, argmax = torch.max(input,2)

--     if self.inplace then
--         self.output = input:zero()
--     else
--         self.output = torch.FloatTensor():resizeAs(input):zero()
--     end
--     local N = input:size(1)
--     -- self.ind must be a tensor of 1 ndimension
--     local spSize = self.output:size(3)*self.output:size(4)
--     self.ind = torch.Tensor(N*spSize):zero()
--     for b=1,N do
--         for i=1,self.output:size(3) do
--             for j=1,self.output:size(4) do
--                 ind = argmax[{b,1,i,j}]
--                 self.output[{b,ind,i,j}] = 1

--                 local tmp = (b-1)*spSize + (i-1)*self.output:size(3)+j
--                 self.ind[tmp] = ind

--             end
--         end
--     end

--     self.output = self.output:type(self._type)

--     if nInputDim == 3 then
--         self.output = self.output[1]
--     end

--     return self.output, self.ind
-- end

function MaxCoord:updateGradInput(input, gradOutput)
    -- do nothing
    assert(1==2)

    return self.gradInput
end