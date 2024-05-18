mutable struct PrisonersDilemmaEnv
    score_agent_1::Int
    score_agent_2::Int
    conditions::Vector{String}

    function PrisonersDilemmaEnv()
        new(0, 0, ["CC", "CD", "DC", "DD"])
    end
end

function trial(env::PrisonersDilemmaEnv, action_1::Int, action_2::Int)

    if action_1 == 1 && action_2 == 1
        obs_1 = 1  # CC for Agent 1
        obs_2 = 1  # CC for Agent 2
        points_1, points_2 = 3, 3  
    elseif action_1 == 1 && action_2 == 2
        obs_1 = 2  # CD for Agent 1
        obs_2 = 3  # DC for Agent 2
        points_1, points_2 = 0, 5  
    elseif action_1 == 2 && action_2 == 1
        obs_1 = 3  # DC for Agent 1
        obs_2 = 2  # CD for Agent 2
        points_1, points_2 = 5, 0  
    elseif action_1 == 2 && action_2 == 2
        obs_1 = 4  # DD for Agent 1
        obs_2 = 4  # DD for Agent 2
        points_1, points_2 = 1, 1 
    end

    env.score_agent_1 += points_1
    env.score_agent_2 += points_2

    return env.conditions[obs_1], env.conditions[obs_2], env.score_agent_1, env.score_agent_2
end