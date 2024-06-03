using Pkg
using Plots

using ActiveInference
using LinearAlgebra

#Pkg.develop(path=raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\Continued_ActiveInference\Dev_Branch\ActiveInference.jl")


###### Defining environment ###########

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

env = PrisonersDilemmaEnv()

########## Creating Agents ##################
conditions = ["CC","CD","DC", "DD"]

states = [4]
observations = [4]
actions = ["COOPERATE", "DEFECT"]

controls = [length(actions)]

A_matrix, B_matrix = generate_random_GM(states, observations, controls)

A_matrix[1] = Matrix{Float64}(I, 4, 4)

########## Creating AIF agent ##############

# When you cooperate, the is a zero probability that the observation will start with D and VV
for i in eachindex(B_matrix)
    B_matrix[i] .= 0.0
end

# other conditions are 50/50
B_matrix[1][1,:,1] = [0.5,0.5,0.5,0.5]
B_matrix[1][2,:,1] = [0.5,0.5,0.5,0.5]

B_matrix[1][3,:,2] = [0.5,0.5,0.5,0.5]
B_matrix[1][4,:,2] = [0.5,0.5,0.5,0.5]

B_matrix[1]

C = array_of_any_zeros(4)
C[1][1] = 3.0 # CC
C[1][2] = 0.0 # CD
C[1][3] = 5.0 # DC
C[1][4] = 1.0 # DD

# Parameterize and keep rations
beta = 10.0
C[1] = softmax(C[1] * beta)
C[1]
pB = deepcopy(B_matrix)

for i in eachindex(pB)
    pB[i] = pB[i] .* 2.0
end

settings=Dict("use_param_info_gain" => true,
              "action_selection" => "stochastic",
              "policy_len" => 4)

parameters = Dict{String, Real}("lr_pB" => 1.0)


AIF_agent = init_aif(A_matrix, B_matrix;
                    C=C,
                    pB=pB,
                    settings=settings,
                    parameters=parameters);



########## Creating TitForTatAgent ##############
mutable struct TitForTatAgent
    last_opponent_action::Vector{Int64}

    function TitForTatAgent()
        new([1])
    end
end

function update_TFT(agent::TitForTatAgent, observation::Vector{Int64})
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

TFT_agent = TitForTatAgent()
########## Trial loop ###########

N_TRIALS = 1500

# Starting Observation
obs1 = [1]
obs2 = [1]

actions_TFT_store = []
actions_AIF_store = []

obs_TFT_store = []
obs_AIF_store = []

for t in 1:N_TRIALS

    infer_states!(AIF_agent, obs1)

    # Taking the first observation as last action
    update_TFT(TFT_agent, obs2)

    # Update Transitions SAMUEL
    if get_states(AIF_agent)["action"] !== missing
        QS_prev = get_history(AIF_agent)["posterior_states"][end-1]
        #AIF_agent.qs_current = process_observation(obs1, length(A_matrix), observations)
        update_B!(AIF_agent, QS_prev)
    end

    infer_policies!(AIF_agent)

    action_AIF_agent = sample_action!(AIF_agent)
    action_AIF_agent = Int(action_AIF_agent[1])
    push!(actions_AIF_store, action_AIF_agent)

    action_TFT_agent = choose_action_TFT(TFT_agent)
    action_TFT_agent = Int(action_TFT_agent[1])
    push!(actions_TFT_store, action_TFT_agent)

    println(" **** At Trial: $(t) **** \n AIF_agent Plays: $(action_AIF_agent) \n TFT_agent Plays: $(action_TFT_agent)")
    obs1, obs2, score_AIF_agent, score_TFT_agent = trial(env, action_AIF_agent, action_TFT_agent)
    obs1 = [findfirst(isequal(obs1), conditions)]
    obs2 = [findfirst(isequal(obs2), conditions)]

    push!(obs_AIF_store, obs1)
    push!(obs_TFT_store, obs2)
    
    println("AIF_agent OBSERVES: $(obs1) \n SCORE AIF_agent: $(score_AIF_agent)")
    println("TFT_agent OBSERVES: $(obs2) \n SCORE TFT_agent: $(score_TFT_agent)")

end

action_matrix = [actions_AIF_store'; actions_TFT_store']

cmap = ["green", "red"]

# Plot the heatmap
heatmap(action_matrix[:,901:1000], color=cmap,
        clims = (1,2),
        legend=false, ylabel="Agent",
        yticks=(1:2, ["AIF", "TFT"]),
        size=(1000, 80))


AIF_agent.B[1]


