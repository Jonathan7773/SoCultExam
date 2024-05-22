using Pkg
using Plots

include(raw"..\AIFInitFunctions.jl")
include(raw"..\AlgoAgentsFunctions.jl")
include(raw"..\GenerativeModel.jl")
include(raw"..\PrisonersDilemmaEnv.jl")

settings=Dict("use_param_info_gain" => false,
              "use_states_info_gain" => false,
              "action_selection" => "deterministic")

# Reinitialize agents
AIF_agent = init_agent_lr_alpha(1.0, 16)



mutable struct TitFor2TatsAgent
    last_opponent_action::Vector{Int64}
    opponent_action_hist::Vector{Int64}

    function TitFor2TatsAgent()
        new([1], [])
    end
end

function update_AlgoAgent(agent::TitFor2TatsAgent, observation::Vector{Int64})

    if observation == [1]
        agent.last_opponent_action = [1]
    elseif observation == [2]
        agent.last_opponent_action = [2]
    elseif observation == [3]
        agent.last_opponent_action = [1]
    else
        agent.last_opponent_action = [2]
    end

    push!(agent.opponent_action_hist, agent.last_opponent_action[1])
end

function choose_action_AlgoAgent(agent::TitFor2TatsAgent)
    n = length(agent.opponent_action_hist)

    if n >= 2 && agent.opponent_action_hist[end] == 2 && agent.opponent_action_hist[end-1] == 2
        return [2]
    else
        return [1]
    end
end

env = PrisonersDilemmaEnv()

Agent_T = TitFor2TatsAgent()

N_TRIALS = 100

# Starting Observation
obs1 = [1]
obs2 = [1]

actions_AIF_store = []
actions_TitFor2Tats_store = []

obs_AIF_store = []
obs_TitFor2Tats_store = []

score_AIF_agent = 0
score_TitFor2Tats_agent = 0


# Run simulation
for t in 1:N_TRIALS

    infer_states!(AIF_agent, obs1)

    # Taking the first observation as last action
    update_AlgoAgent(Agent_T, obs2)

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

    action_TitFor2Tats_agent = choose_action_AlgoAgent(Agent_T)
    action_TitFor2Tats_agent = Int(action_TitFor2Tats_agent[1])
    push!(actions_TitFor2Tats_store, action_TitFor2Tats_agent)

    println(" **** At Trial: $(t) **** \n AIF_agent Plays: $(action_AIF_agent) \n TitFor2Tats_agent Plays: $(action_TitFor2Tats_agent)")
    obs1, obs2, score_AIF_agent, score_TitFor2Tats_agent = trial(env, action_AIF_agent, action_TitFor2Tats_agent)
    obs1 = [findfirst(isequal(obs1), conditions)]
    obs2 = [findfirst(isequal(obs2), conditions)]

    push!(obs_AIF_store, obs1)
    push!(obs_TitFor2Tats_store, obs2)
    
    println("AIF_agent OBSERVES: $(obs1) \n SCORE AIF_agent: $(score_AIF_agent)")
    println("TitFor2Tats_agent OBSERVES: $(obs2) \n SCORE TitFor2Tats_agent: $(score_TitFor2Tats_agent)")

end

action_matrix = [actions_AIF_store'; actions_TitFor2Tats_store']

cmap = ["green", "red"]

# Plot the heatmap
heatmap(action_matrix, color=cmap,
        clims = (1,2),
        legend=false, ylabel="Agent",
        yticks=(1:2, ["AIF", "TitFor2TatsAgent"]),
        size=(1000, 80))
