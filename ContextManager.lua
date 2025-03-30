--[[
	torch internal module
	
	used to keep track of contexts and check whether code is being ran in a specific context
]]

local context_states = {} --stores all contexts


local ContextManager = {}


--[[
	Enter <strong>name</strong> context
]]
function ContextManager.enter_context(name: string)
	if context_states[name] == nil then
		context_states[name] = {}
	end
	
	context_states[name][coroutine.running()] = true
end

--[[
	Exit <strong>name</strong> context
]]
function ContextManager.exit_context(name: string)
	context_states[name][coroutine.running()] = nil
	
	if #context_states[name] == 0 then --if there's no more threads in this context, remove it from the states table
		context_states[name] = nil
	end
end

--[[
	Check if the current thread is in <strong>name</strong> context
]]
function ContextManager.in_context(name: string)
	if context_states[name] == nil then return false end
	
	return context_states[name][coroutine.running()] == true
end


return ContextManager