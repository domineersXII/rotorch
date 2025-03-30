--[[
	====== RoTorch ======

	PyTorch style machine learning library for Roblox
	
	Author: domineersXII
	Last modified: 3/14/2025
	Version: 1.0.0
	Documentation: LINK GOES HERE!
]]


--MODULES
local nn = require(script.nn)
local autograd = require(script.autograd)
local optim = require(script.optim)

local tensor = require(script.tensor)
local types = require(script.types)
local kwargs = require(script.kwargs)
local ContextManager = require(script.ContextManager)
local getTableSize = require(script.tensor.helpers.getTableSize)
local createTensor = require(script.createTensor)
local functions = require(script.functions)

local torch = {
	nn = nn,
	autograd = autograd,
	optim = optim,
}

type data = {data | number}

local function randomNormal(mean: number, standardDeviation: number)
	local u1 = math.random()
	local u2 = math.random()
	local z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
	
	return mean + z0 * standardDeviation
end


local function loadLargeStringData(data: string, module: ModuleScript)	
	local ScriptEditorService = game:GetService("ScriptEditorService")
	module.Source = "return "

	ScriptEditorService:OpenScriptDocumentAsync(module)
	local doc = ScriptEditorService:FindScriptDocument(module)
	
	print("Yielding for safety, please wait...")
	task.wait(1)
	doc:EditTextAsync(data, 1, 8, 1, 8 + #data)
end


local function typeCheck(input: types.Tensor, other: types.Tensor | number, op: string)
	if type(input) ~= "table" or input._istensor == nil then error(`input 1 of torch.{op} must be an instance of the tensor class.`) end
	if type(other) ~= "table" and type(other) ~= "number" then error(`input 2 of torch.{op} must be an instance of the tensor class, or of type number.`) end
end

local function dataToStringTable(input: data): string
	local str = ""
	
	if type(input[1]) == "table" then
		str ..= "{\n"
		
		for i = 1, #input do
			str ..= dataToStringTable(input[i])
			if i < #input then
				str = str .. ",\n"
			end
		end

		str ..= "\n" .. "}"
	else
		str = str .. "{"

		for i = 1, #input do
			str ..= input[i] 

			if i < #input then
				str ..= ","
			end
		end

		str ..= "}"
	end
	
	return str
end

local function getMax(input: any): number
	local max_value = -math.huge

	local function recursiveMax(data: any)
		if type(data) == "number" then
			if data > max_value then
				max_value = data
			end
		elseif type(data) == "table" then
			for i, v in pairs(data) do
				recursiveMax(v)
			end
		end
	end

	recursiveMax(input)
	return max_value
end


--[[
	Creates a one dimensional tensor of <strong>size</strong> steps whose values are evenly spaced from <strong>start_num</strong> to <strong>end_num</strong>, inclusive.
]]
function torch.linspace(start_num: number, end_num: number, steps: number, kwargs: kwargs.requires_grad): types.Tensor
	if steps == 1 then return tensor._new({start_num}, {1}, kwargs and (kwargs.requires_grad or false) or false, true) end
	
	local data = {}
	local step_size = (end_num - start_num) / (steps - 1)
	
	for i = 0, steps - 1 do
		local value = start_num + i * step_size
		table.insert(data, value)
	end
	
	return tensor._new(data, {#data}, kwargs and (kwargs.requires_grad or false) or false, true)
end

--[[
	Constructs a tensor with input <strong>data</strong>.
]]
function torch.tensor(data: {}, kwargs: kwargs.requires_grad): types.Tensor
	return tensor._new(data, getTableSize(data), kwargs and (kwargs.requires_grad or false) or false, true)
end

--[[
	Returns a tensor of inputted dimensions filled with the value 0.
]]
function torch.zeros(size: {number}, kwargs: kwargs.requires_grad): types.Tensor
	return tensor._new(createTensor(size, 1, function() return 0 end), size, kwargs and (kwargs.requires_grad or false) or false, true)
end

--[[
	Returns a tensor of the same size as input, except filled with 0s.
]]
function torch.zeros_like(input: types.Tensor, kwargs: kwargs.requires_grad): types.Tensor
	return tensor._new(createTensor(input.shape, 1, function() return 0 end), table.clone(input.shape), kwargs and (kwargs.requires_grad or false) or false, true)
end

--[[
	Returns a tensor of inputted dimensions filled with the value 1.
]]
function torch.ones(size: {number}, kwargs: kwargs.requires_grad): types.Tensor
	return 	tensor._new(createTensor(size, 1, function() return 1 end), size, kwargs and (kwargs.requires_grad or false) or false, true)
end

--[[
	Returns a tensor of inputted dimensions filled with random numbers from a uniform distribution on the interval [0,1).
]]
function torch.rand(size: {number}, kwargs: kwargs.requires_grad): types.Tensor
	return tensor._new(createTensor(size, 1, function() return math.random() end), size, kwargs and (kwargs.requires_grad or false) or false, true)
end

--[[
	Returns a tensor of inputted dimensions filled with numbers from a normal distribution with mean 0 and variance 1.
]]
function torch.randn(size: {number}, kwargs: kwargs.requires_grad): types.Tensor
	return tensor._new(createTensor(size, 1, function() return randomNormal(0, 1) end), size, kwargs and (kwargs.requires_grad or false) or false, true)
end

--[[
	Returns a tensor of inpuuted dimensions filled with fill value.
]]
function torch.full(size: {number}, fill: number, kwargs: kwargs.requires_grad): types.Tensor
	return tensor._new(createTensor(size, 1, function() return fill end), size, kwargs and (kwargs.requires_grad or false) or false, true)
end

--[[
	Returns a new tensor with values that equal the sum of (<strong>input</strong> + <strong>other</strong>).
]]
function torch.add(input: types.Tensor, other: types.Tensor | number): types.Tensor
	typeCheck(input, other, "add")
	
	return tensor.__add(input, other)
end

--[[
	Returns a new tensor with values that equal the difference of the (<strong>input</strong> - <strong>other</strong>).
]]
function torch.sub(input: types.Tensor, other: types.Tensor | number): types.Tensor
	typeCheck(input, other, "sub")
	
	return tensor.__sub(input, other)
end

--[[
	Returns a new tensor with values that equal the product of (<strong>input</strong> * <strong>other</strong>).
]]
function torch.mul(input: types.Tensor, other: types.Tensor | number): types.Tensor
	typeCheck(input, other, "mul")
	
	return tensor.__mul(input, other)
end

--[[
	Returns a new tensor with values that equal the quotient of (<strong>input</strong> / <strong>other</strong>).
]]
function torch.div(input: types.Tensor, other: types.Tensor | number): types.Tensor
	typeCheck(input, other, "div")
	
	return tensor.__div(input, other)
end

function torch.pow(input: types.Tensor, other: types.Tensor | number): types.Tensor
	typeCheck(input, other, "pow")
	
	return tensor.__pow(input, other)
end

--function torch.sum()
	--there is an internal implementation(torch -> tensor -> helpers -> sum) but its kind of bad and its non differentiable
--end

--[[
	Returns a new tensor with the same data as the input tensor but of a different shape.
]]
function torch.view(input: types.Tensor, ...: number): types.Tensor
	return functions.view.apply(input, ...)
end

--[[
	Performs the element-wise division of tensor1 by tensor2, multiplies the result by the scalar value and adds it to input.
]]
function torch.addcdiv(input: types.Tensor, tensor1: types.Tensor, tensor2: types.Tensor, value: number?): types.Tensor
	return functions.addcdiv.apply(input, tensor1, tensor2, value or 1)
end

--[[
	Returns a scalar tensor consisting of the maximum value in the input tensor.
]]
function torch.max(input: types.Tensor): types.Tensor
	return tensor._new({getMax(input.data)}, {1}, input.requires_grad, true)
end

--[[
	Performs a matrix multiplication of the matrices <strong>input</strong> and <strong>mat2</strong>.
	If input is a <em><strong>(n×m)</em></strong> tensor, mat2 is a <em><strong>(m×p)</em></strong> tensor, out will be a <em><strong>(n×p)</em></strong> tensor.
]]
function torch.mm(input: types.Tensor, mat2: types.Tensor): types.Tensor
	assert(type(input) == "table" and input._istensor, "input 1 of torch.mm must be an instance of the tensor class.")
	assert(type(mat2) == "table" and mat2._istensor, "input 2 of torch.mm must be an instance of the tensor class.")
	 
	return functions.mm.apply(input, mat2)
end

--[[
	Returns the value of a scalar tensor as a number.
]]
function torch.item(input: types.Tensor): number
	assert(#input.shape == 1 and input.shape[1] == 1, "torch.item expects a scalar tensor as its input.")
	
	return input.data[1]
end

--[[
	Performs an elementwise square root operation on the inputted tensor.
]]
function torch.sqrt(input: types.Tensor): types.Tensor
	return functions.sqrt.apply(input)
end

--[[
	Run a piece of code in the no_grad context (gradients will not be calculated here, and all tensors created in the no_grad context will have requires_grad=false)
]]
function torch.no_grad(fn: () -> ())
	ContextManager.enter_context("no_grad")
	fn()
	ContextManager.exit_context("no_grad")
end

--[[
	Saves a tensor or table of tensors to a folder of module scripts with the inputted name
	
	<strong>[MUST BE RAN IN PLUGIN CONTEXT]</strong>
]]
function torch.save(obj: types.Tensor | {types.Tensor}, name: string?)
	if plugin == nil then error("torch.save can only be ran in a plugin context.") end
	
	local folder = Instance.new("Folder")
	folder.Name = `{if name then name else "stored"}_folder_rdata`
	folder.Parent = game.ServerStorage

	if typeof(obj) == "table" and obj._istensor ~= true then
		for i, tensor in obj do
			local string_data = dataToStringTable(tensor.data)
			local module = Instance.new("ModuleScript")
			module.Name = `tensor{i}_rdata`
			module:SetAttribute("id", i)
			module.Parent = folder
			
			if #string_data >= 199_900 then
				loadLargeStringData(string_data, module)
				continue
			end
			
			module.Source = "return " .. string_data
		end
	else
		local string_data = dataToStringTable(obj.data)
		local module = Instance.new("ModuleScript")
		module.Name = `tensor1_rdata`
		module:SetAttribute("id", 1)
		module.Parent = folder
		
		if #string_data >= 199_900 then
			loadLargeStringData(string_data, module)
			return
		end
		
		module.Source = "return " .. string_data
	end
end

--[[
	Loads tensor(s) from a folder of _rdata modules and returns a table of the loaded data as tensors.
]]
function torch.load(location: Folder, kwargs: kwargs.requires_grad)
	--sort data in the correct order
	local module_data = {}
	
	for i, child in location:GetChildren() do
		if child:IsA("ModuleScript") == false then warn("Child that is not a module script found in an rdata folder.") continue end
		
		table.insert(module_data, child)
	end
	
	table.sort(module_data, function(a: ModuleScript, b: ModuleScript)
		return a:GetAttribute("id") < b:GetAttribute("id")
	end)
	
	--require and convert data to tensors
	local loaded_data = {}
	
	for i, module in module_data do
		local data = require(module)
		table.insert(loaded_data, torch.tensor(data, if kwargs and kwargs.requires_grad then kwargs.requires_grad else false))
	end
	
	return loaded_data
end

return torch