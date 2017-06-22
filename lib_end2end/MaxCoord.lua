--[[
Expects 3D or 4D input. Does a max over the feature channels.
--]]
local MaxCoord, parent = torch.class('nn.MaxCoord', 'nn.Module')

function MaxCoord:__init(inplace)
    parent.__init(self)
    self.inplace = inplace or false

    -- added by yan
    -- self.maxPos = torch.Tensor()

end

function MaxCoord:updateOutput(input)
    local nInputDim = input:nDimension()
    if input:nDimension() == 3 then
        local C,H,W = input:size(1), input:size(2), input:size(3)
        input = input:view(1,C,H,W)
    end
    assert(input:nDimension()==4, 'Input must be 3D or 4D (batch).')
    -- print('self._type '.. self._type)
    if self._type ~= 'torch.FloatTensor' then
        input = input:float()
    end
    -- print('self._type '.. self._type)
    -- Ab()
    input = input:float()

    local _, argmax = torch.max(input,2)

    if self.inplace then
        self.output = input:zero()
    else
        self.output = torch.FloatTensor():resizeAs(input):zero()
    end
    local N = input:size(1)
    -- self.ind must be a tensor of 1 ndimension
    local spSize = self.output:size(3)*self.output:size(4)
    self.ind = torch.Tensor(N*spSize):zero()
    for b=1,N do
        for i=1,self.output:size(3) do
            for j=1,self.output:size(4) do
                ind = argmax[{b,1,i,j}]
                self.output[{b,ind,i,j}] = 1

                local tmp = (b-1)*spSize + (i-1)*self.output:size(3)+j
                self.ind[tmp] = ind

            end
        end
    end

    self.output = self.output:type(self._type)

    if nInputDim == 3 then
        self.output = self.output[1]
    end

    -- -- added by Yan
    -- self.maxPos = self.output
    return self.output, self.ind
end

-- The original version doesn't need to backward over swap, so just use 0s as its value.

-- function MaxCoord:updateGradInput(input, gradOutput)
--     self.gradInput:resizeAs(input):zero()
--     return self.gradInput
-- end

-- function MaxCoord:updateGradInput(input, gradOutput)
--     -- Here gradOutput and self.maxPos should be bs*C*H*w, So use cmul
--     -- because the self.maxPos has been view as 4 dimensions, so gradOutput should be the same size.
--     assert(self.maxPos:nDimension() == gradOutput:nDimension(),'gradInput dimension should be the same as that of the input')
--     local bs,C,H,W = self.maxPos:size(1), self.maxPos:size(2), self.maxPos:size(3), self.maxPos:size(4)
--     self.maxPos = self.maxPos:view(-1)
--     gradOutput = gradOutput:view(-1)
--     local tmp = torch.Tensor(bs*C*H*W):type(self._type)  -- This kind way is excellent!


--     torch.cmul(tmp, self.maxPos, gradOutput)
--     self.gradInput = tmp:resize(bs,C,H,W)

--     return self.gradInput
-- end

function MaxCoord:updateGradInput(input, gradOutput)
    -- do nothing
    assert(1==2)

    return self.gradInput
end