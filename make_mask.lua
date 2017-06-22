require 'torch'
require 'lib/myUtils'
require 'image' 

opt = {
	fineSize = 256,
	test = 0,
	batchSize = 1,
}

print(opt)

local mask_global = torch.ByteTensor(opt.batchSize, opt.fineSize, opt.fineSize)

local res = 0.05 -- the lower it is, the more continuous the output will be. 0.01 is too small and 0.1 is too large
local density = 0.30
local MAX_SIZE = 10000
local low_pattern = torch.Tensor(res*MAX_SIZE, res*MAX_SIZE):uniform(0,1):mul(255)
local pattern = image.scale(low_pattern, MAX_SIZE, MAX_SIZE,'bicubic')
low_pattern = nil
pattern:div(255);
pattern = torch.lt(pattern,density):byte()  -- 25% 1s and 75% 0s
pattern = pattern:byte()
print('...Random pattern generated')

for i = 1,2100 do

mask_global =  create_gMask(pattern, mask_global, MAX_SIZE, opt)

--print(#mask_global)

local mask_temp = mask_global:squeeze()

print(torch.sum(mask_temp)/65535)

image.save(paths.concat('./mask_val/',i..'.png'),mask_temp:mul(255))

end
