using ActiveInference

mutable struct TitForTatAgent
    last_opponent_action::Vector{Int64}

    function TitForTatAgent()
        new([1])
    end
end

function update_AlgoAgent(agent::TitForTatAgent, observation::Vector{Int64})
    if observation == [1]
        agent.last_opponent_action = [1]
    elseif observation == [2]
        agent.last_opponent_action = [2]
    elseif observation == [3]
        agent.last_opponent_action = [1]
    else
        agent.last_opponent_action = [2]
    end
end

function choose_action_TFT(agent::TitForTatAgent)
    return agent.last_opponent_action
end

function init_agent_lr_alpha(lr_AIF, alpha_AIF)
    parameters_AIF = Dict{String, Real}("lr_pB" => lr_AIF,
                                            "alpha" => alpha_AIF)

    AIF_agent = init_aif(A_matrix, B_matrix;
                      C=C,
                      pB=pB,
                      settings=settings,
                      parameters=parameters_AIF,
                      verbose = false)

    return AIF_agent
end


function init_agent_lr_beta(lr_AIF, beta_AIF)
    parameters_AIF = Dict{String, Real}("lr_pB" => lr_AIF,
                                            "alpha" => 16)
    C = array_of_any_zeros(4)
    C[1][1] = 3.0 # CC
    C[1][2] = 1.0 # CD
    C[1][3] = 4.0 # DC
    C[1][4] = 2.0 # DD

    β = beta_AIF
    C[1] = softmax(C[1] * β)

    AIF_agent = init_aif(A_matrix, B_matrix;
                      C=C,
                      pB=pB,
                      settings=settings,
                      parameters=parameters_AIF,
                      verbose = false)

    return AIF_agent
end


function init_AIF_agents(lr_samuel, lr_jonathan)
    parameters_samuel = Dict{String, Real}("lr_pB" => lr_samuel,
                                            "alpha" => 16)
    parameters_jonathan = Dict{String, Real}("lr_pB" => lr_jonathan,
                                            "alpha" => 16)

    SAMUEL = init_aif(A_matrix, B_matrix;
                      C=C,
                      pB=pB,
                      settings=settings,
                      parameters=parameters_samuel,
                      verbose = false)

    JONATHAN = init_aif(A_matrix, B_matrix;
                        C=C,
                        pB=pB,
                        settings=settings,
                        parameters=parameters_jonathan,
                        verbose = false)

    return SAMUEL, JONATHAN
end

function init_agent_lr(lr_AIF)
    parameters_AIF = Dict{String, Real}("lr_pB" => lr_AIF,
                                            "alpha" => 16)

    AIF_agent = init_aif(A_matrix, B_matrix;
                      C=C,
                      pB=pB,
                      settings=settings,
                      parameters=parameters_AIF,
                      verbose = false)

    return AIF_agent
end

