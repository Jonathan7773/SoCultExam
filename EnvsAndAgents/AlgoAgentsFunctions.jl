############# Agent Structs ###########

""" TitForTat Agent """
mutable struct TitForTatAgent
    last_opponent_action::Vector{Int64}

    function TitForTatAgent()
        new([1])
    end
end

""" TitFor2Tats Agent """
mutable struct TitFor2TatsAgent
    last_opponent_action::Vector{Int64}
    opponent_action_hist::Vector{Int64}

    function TitFor2TatsAgent()
        new([1], [])
    end
end

""" TwoTitsFor1Tat Agent """
mutable struct TwoTitsFor1TatAgent
    last_opponent_action::Vector{Int64}
    opponent_action_hist::Vector{Int64}
    punishment_count::Int64

    function TwoTitsFor1TatAgent()
        new([1], [], 0)
    end
end

""" NastyForgivingTFT Agent """
mutable struct NastyForgivingTFTAgent
    last_opponent_action::Vector{Int64}
    opponent_action_hist::Vector{Int64}

    function NastyForgivingTFTAgent()
        new([2], [])
    end
end

""" Pavlovian Agent """
mutable struct PavlovianAgent
    last_action::Vector{Int64}
    last_opponent_action::Vector{Int64}
    last_outcome::Vector{Int64} 

    function PavlovianAgent()
        new([1], [1], [1])
    end
end

""" Grofman Agent """
mutable struct GrofmanAgent
    last_opponent_action::Vector{Int64}
    opponent_action_hist::Vector{Int64}
    last_own_action::Vector{Int64}
    own_action_hist::Vector{Int64}

    function GrofmanAgent()
        new([1], [], [],[])
    end
end

""" GrimTrigger Agent """
mutable struct GrimTriggerAgent
    last_opponent_action::Vector{Int64}
    defected::Bool

    function GrimTriggerAgent()
        new([1], false)
    end
end

############# Update Functions ################

""" TitForTat Agent """
function update_AlgoAgent(agent::TitForTatAgent, observation::Vector{Int64})
    if observation == [1] || observation == [3]
        agent.last_opponent_action = [1]
    else
        agent.last_opponent_action = [2]
    end
end

""" TitFor2Tats Agent """
function update_AlgoAgent(agent::TitFor2TatsAgent, observation::Vector{Int64})

    if observation == [1] || observation == [3]
        agent.last_opponent_action = [1]
    else
        agent.last_opponent_action = [2]
    end

    push!(agent.opponent_action_hist, agent.last_opponent_action[1])
end

""" TwoTitsFor1Tat Agent """
function update_AlgoAgent(agent::TwoTitsFor1TatAgent, observation::Vector{Int64})

    if observation == [1] || observation == [3]
        agent.last_opponent_action = [1]
    else
        agent.last_opponent_action = [2]
    end

    push!(agent.opponent_action_hist, agent.last_opponent_action[1])

    if agent.last_opponent_action == [2]
        agent.punishment_count = 2
    elseif agent.punishment_count > 0
        agent.punishment_count -= 1
    end

end

""" NastyForgivingTFT Agent """
function update_AlgoAgent(agent::NastyForgivingTFTAgent, observation::Vector{Int64})

    if observation == [1] || observation == [3]
        agent.last_opponent_action = [1]
    else
        agent.last_opponent_action = [2]
    end

    push!(agent.opponent_action_hist, agent.last_opponent_action[1])
end

""" Pavlovian Agent """
function update_AlgoAgent(agent::PavlovianAgent, observation::Vector{Int64})

    if observation == [1] || observation == [2]
        agent.last_action = [1]
    else
        agent.last_action = [2]
    end

    if observation == [1] || observation == [3]
        agent.last_outcome = [1]
    else
        agent.last_outcome = [0]
    end

end

""" Grofman Agent """
function update_AlgoAgent(agent::GrofmanAgent, observation::Vector{Int64})
    
    if observation == [1] || observation == [3]
        agent.last_opponent_action = [1]
    else
        agent.last_opponent_action = [2]
    end

    if observation == [1] || observation == [2]
        agent.last_own_action = [1]
    else
        agent.last_own_action = [2]
    end

    push!(agent.opponent_action_hist, agent.last_opponent_action[1])
    push!(agent.own_action_hist, agent.last_own_action[1])

end

""" GrimTrigger Agent """
function update_AlgoAgent(agent::GrimTriggerAgent, observation)

    if observation == [1] || observation == [3]
        agent.last_opponent_action = [1]
    else
        agent.last_opponent_action = [2]
    end

    if agent.last_opponent_action == [2]
        agent.defected = true
    end
end

############# Action Selection ####################

""" TitForTat Agent """
function choose_action_AlgoAgent(agent::TitForTatAgent)
    return agent.last_opponent_action
end

""" TitFor2Tats Agent """
function choose_action_AlgoAgent(agent::TitFor2TatsAgent)
    n = length(agent.opponent_action_hist)

    if n >= 2 && agent.opponent_action_hist[end] == 2 && agent.opponent_action_hist[end-1] == 2
        return [2]
    else
        return [1]
    end
end

""" TwoTitsFor1Tat Agent """
function choose_action_AlgoAgent(agent::TwoTitsFor1TatAgent)
    if agent.punishment_count > 0
        return [2]
    else
        return [1]
    end
end

""" NastyForgivingTFT Agent """
function choose_action_AlgoAgent(agent::NastyForgivingTFTAgent)
    defect_count = count(x -> x == 2, agent.opponent_action_hist)

    if defect_count == 1
        return [1]
    elseif length(agent.opponent_action_hist) == 1
        return [2]
    else
        return agent.last_opponent_action
    end
end

""" Pavlovian Agent """
function choose_action_AlgoAgent(agent::PavlovianAgent)
    if agent.last_outcome == [1]
        return agent.last_action
    else
        if agent.last_action == [1]
            agent.last_action = [2]
        else
            agent.last_action = [1]
        end
        return agent.last_action
    end
end

""" Grofman Agent """
function choose_action_AlgoAgent(agent::GrofmanAgent)
    n = length(agent.opponent_action_hist)

    if n <= 2
        return [1]
    elseif n <= 7
        action = agent.last_opponent_action
    else
        recent_opponent_actions = agent.opponent_action_hist[end-7:end-1]
        recent_defections = count(x -> x == 2, recent_opponent_actions)

        if agent.own_action_hist[end] == 1
            if recent_defections < 3
                action = [1]
            else
                action = [2]
            end
        else
            if recent_defections <= 1
                action = [1]
            else
                action = [2]
            end
        end
    end

    return action
end

""" GrimTrigger Agent """
function choose_action_AlgoAgent(agent::GrimTriggerAgent)
    if agent.defected
        return [2]
    else
        return [1]
    end
end

