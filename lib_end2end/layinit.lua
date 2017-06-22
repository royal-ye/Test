require 'nn'
local layinit = {}



-- latter_init is a switch deciding whether the latter part of Unet should be init with that of the encDec latter part.
-- bias is useless.
function layinit.encDec_init_Unet(encDec, Unet, latter_init)
    local latter_init = latter_init or false
    local tyName
    if torch.type(encDec) == 'nn.Sequential' then
    -- first extract encDec.modules[1]
        local net_first = encDec.modules[1]
        local k = 0
        for i = 1, net_first:size(1) do
            tyName = torch.type(net_first.modules[i])
            print(tyName)
            if tyName == 'cudnn.SpatialConvolution' or tyName == 'nn.SpatialConvolution' then
                print(string.format('asssigning the weight of layer %d to Unet %d', i, i))
                Unet.modules[i].weight = net_first.modules[i].weight:clone()
            end 
            k = k + 1
        end
        local eIdx = 2
        local uIdx = k
        if latter_init then
        -- second assign other layer weight of encDec to unet
            for eIdx = 2, encDec:size(1) do
                tyName = torch.type(encDec.modules[eIdx])
                if tyName == 'cudnn.SpatialFullConvolution' or tyName == 'nn.SpatialFullConvolution' then
                    while(true) do
                        uIdx = uIdx + 1
                        tyName_u = torch.type(Unet.modules[uIdx])
                        if tyName_u == 'cudnn.SpatialFullConvolution' or tyName_u == 'nn.SpatialFullConvolution' then

                            local wt_sz = encDec.modules[eIdx].weight:size()
                            -- local bias_sz = encDec.modules[eIdx].bias:size()
                            Unet.modules[uIdx].weight[{{1,wt_sz[1]},{1,wt_sz[2]},{},{}}] = encDec.modules[eIdx].weight:clone()
                            -- Unet.modules[uIdx].bias[{{1,bias_sz[1]}}] = encDec.modules[eIdx].bias:clone()

                            print(string.format('asssigning the weight of layer %d to Unet layer %d', eIdx, uIdx))
                            break
                        else
                            print(string.format('For enc-dec layer %d: skipping unet layer %d, name:%s', eIdx, uIdx, torch.type(Unet.modules[uIdx])))
                        end
                    end
                else
                    print(string.format('Skipping enc-dec layer %d, name:%s', eIdx, torch.type(encDec.modules[eIdx])))
                    print()
                end
            end
        end
    elseif torch.type(encDec) == 'nn.gModule' then
        local eIdx = 0
        local uIdx = 0
        local tyName, tyName_u = nil, nil
        for eIdx = 1, encDec:size(1) do
            tyName = torch.type(encDec.modules[eIdx])
            if tyName == 'cudnn.SpatialConvolution' or tyName == 'nn.SpatialConvolution' then
                while(true) do
                    uIdx = uIdx + 1
                    tyName_u = torch.type(Unet.modules[uIdx])
                    if tyName_u == 'cudnn.SpatialConvolution' or tyName_u == 'nn.SpatialConvolution' then
                        Unet.modules[uIdx].weight = encDec.modules[eIdx].weight:clone()
                        -- Unet.modules[uIdx].bias = encDec.modules[eIdx].bias:clone()

                        print(string.format('asssigning the weight of layer %d to Unet layer %d', eIdx, uIdx))
                        break
                    else
                        print(string.format('For enc-dec layer %d: skipping unet layer %d, name:%s', eIdx, uIdx, torch.type(Unet.modules[uIdx])))
                    end
                end
            elseif tyName == 'cudnn.SpatialFullConvolution' or tyName == 'nn.SpatialFullConvolution' then
                if latter_init then
                    while(true) do
                        uIdx = uIdx + 1
                        tyName_u = torch.type(Unet.modules[uIdx])
                        if tyName_u == 'cudnn.SpatialFullConvolution' or tyName_u == 'nn.SpatialFullConvolution' then

                            local wt_sz = encDec.modules[eIdx].weight:size()
                            -- local bias_sz = encDec.modules[eIdx].bias:size()
                            Unet.modules[uIdx].weight[{{1,wt_sz[1]},{1,wt_sz[2]},{},{}}] = encDec.modules[eIdx].weight:clone()
                            -- Unet.modules[uIdx].bias[{{1,bias_sz[1]}}] = encDec.modules[eIdx].bias:clone()

                            print(string.format('asssigning the weight of layer %d to Unet layer %d', eIdx, uIdx))
                            break
                        else
                            print(string.format('For enc-dec layer %d: skipping unet layer %d, name:%s', eIdx, uIdx, torch.type(Unet.modules[uIdx])))
                        end
                    end
                end
            else
                print(string.format('Skipping enc-dec layer %d, name:%s', eIdx, torch.type(encDec.modules[eIdx])))
                print()
            end
        end
    else
        error('wrong module type!')
    end
    return Unet

end



return layinit