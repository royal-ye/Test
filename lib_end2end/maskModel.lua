-- In this script, try to swap in just one forward!

-- No latent, just use input is enough!
-- DO NOT get self.mask in __init, just extract it from input, so that the MaskModel layer
-- can just create one time and use it all the process!

local MaskModel, parent = torch.class('nn.MaskModel', 'nn.Module')

function MaskModel:__init(conv_layers, threshold)
    parent.__init(self)
    self.threshold = threshold
    self.conv_layers = conv_layers
    self.lnet = nn.Sequential()
    -- local preLayer = nn.SpatialConvolution(1, 1, 4, 4, 2, 2, 1, 1):noBias()
    -- preLayer.weight:fill(1.0/16)
    -- lent = - preLayer
    for id_net = 1, self.conv_layers do
        local conv = nn.SpatialConvolution(1, 1, 4, 4, 2, 2, 1, 1):noBias()
        conv.weight:fill(1/16)
        self.lnet = self.lnet:add(conv)
    end

-- A quick fix, need to update!!!!
    -- default conv_layers == 2
    assert(self.conv_layers == 2,'conv_layers has to equal to 2, need improve!')
    -- local conv1 = nn.SpatialConvolution(1, 1, 4, 4, 2, 2, 1, 1):noBias()
    -- conv1.weight:fill(1/16)
    -- local conv2 = nn.SpatialConvolution(1, 1, 4, 4, 2, 2, 1, 1):noBias()
    -- conv2.weight:fill(1/16)
    -- local e1 = - conv1
    -- local e2 = e1 - conv2
    -- self.lnet = nn.gModule({e1},{e2})

    print('lnet')
    print(self.lnet)
  --  printNet(self.lnet)
end


function MaskModel:updateOutput(input)
    assert(input:nDimension() == 3)  -- or 4 ??
    self.output = self.lnet:forward(input)
    local fLen = self.output:size(2)
    local i,j
    for i = 1, fLen do
        for j = 1, fLen do
            local v = torch.mean(self.output[{{i},{j}}])
            self.output[{{},{i},{j}}] = v > self.threshold and 1 or 0
        end
    end    

    return self.output
end


-- It can be set to anything, it doesn't matter.
function MaskModel:updateGradInput(input, gradOutput)
    self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
    self.gradInput = nn.utils.recursiveFill(self.gradInput, 0)
    return self.gradInput

end


function MaskModel:clearState()
    self.output = self.output.new()
    self.gradInput = self.gradInput.new()

end

